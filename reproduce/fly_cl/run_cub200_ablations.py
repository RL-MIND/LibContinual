"""Run the Fly-CL CUB-200 component ablations and sensitivity sweeps.

The frozen ViT features are cached once per input-normalization setting.  Every
analytic experiment still uses ``core.model.fly_cl.FlyCL`` for projection,
streaming statistics, GCV, Cholesky solving, prototype construction, and
inference.  This removes repeated backbone inference without changing the
algorithm under study.
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from core.data import get_dataloader
from core.model.backbone.timm_backbone import timm_backbone
from core.model.fly_cl import FlyCL
from core.utils import init_seed


COMPONENT_EXPERIMENTS = [
    {"name": "full", "normalization": True},
    {
        "name": "wo_projection",
        "normalization": True,
        "use_projection": False,
        # Removing the expansion changes the feature scale by orders of
        # magnitude.  A broad GCV grid is required for a meaningful ablation.
        "ridge_lower": 0,
        "ridge_upper": 10,
    },
    {"name": "wo_ridge", "normalization": True, "use_ridge": False},
    {"name": "wo_normalization", "normalization": False},
    {
        "name": "wo_all",
        "normalization": False,
        "use_projection": False,
        "use_ridge": False,
    },
]

SENSITIVITY_EXPERIMENTS = (
    [
        {"name": f"m_{m}", "group": "expand_dim", "expand_dim": m}
        for m in (500, 1000, 2000, 5000, 10000, 20000)
    ]
    + [
        {"name": f"p_{p}", "group": "synaptic_degree", "synaptic_degree": p}
        for p in (50, 100, 200, 300, 500, 700, 768)
    ]
    + [
        {
            "name": f"k_{k}",
            "group": "activation_sparsity",
            "coding_level": k / 10000,
            "coding_k": k,
        }
        for k in (500, 1000, 2000, 3000, 4000, 5000, 7000, 8000, 9000, 10000)
    ]
)

EXPECTED_EXPERIMENTS = {
    spec["name"] for spec in COMPONENT_EXPERIMENTS + SENSITIVITY_EXPERIMENTS
}


class IdentityBackbone(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, features):
        return features


def _without_normalization(config):
    config = copy.deepcopy(config)
    for key in ("train_trfms", "test_trfms"):
        config[key] = [item for item in config[key] if "Normalize" not in item]
    return config


@torch.no_grad()
def _extract_split(backbone, loaders, device):
    tasks = []
    for task_idx, loader in enumerate(loaders.dataloaders):
        features, labels = [], []
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features.append(backbone(images).cpu())
            labels.append(batch["label"].cpu())
        tasks.append(
            {
                "features": torch.cat(features),
                "labels": torch.cat(labels),
                "task": task_idx,
            }
        )
    return tasks


def build_feature_cache(config, normalized, cache_path, device):
    variant = config if normalized else _without_normalization(config)
    init_seed(variant["seed"], variant["deterministic"])
    backbone = timm_backbone(**variant["backbone"]["kwargs"]).to(device).eval()

    # FlyCL constructs its random projection immediately after the backbone.
    # Saving this state lets cached-feature experiments reproduce that ordering.
    projection_rng_state = torch.get_rng_state().clone()
    train_loaders = get_dataloader(variant, "train")
    test_loaders = get_dataloader(
        variant, "test", cls_map=train_loaders.cls_map
    )
    started = time.perf_counter()
    train = _extract_split(backbone, train_loaders, device)
    test = _extract_split(backbone, test_loaders, device)
    cache = {
        "normalized": normalized,
        "seed": variant["seed"],
        "class_map": train_loaders.cls_map,
        "projection_rng_state": projection_rng_state,
        "train": train,
        "test": test,
        "feature_seconds": time.perf_counter() - started,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    del backbone, train_loaders, test_loaders
    torch.cuda.empty_cache()
    return cache


def load_or_build_cache(config, normalized, cache_path, device, rebuild=False):
    if cache_path.exists() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("seed") != config["seed"]:
            raise ValueError(f"Stale cache seed in {cache_path}")
        return cache
    return build_feature_cache(config, normalized, cache_path, device)


@torch.no_grad()
def evaluate_task(model, task, device, batch_size=512):
    correct = 0
    total = len(task["labels"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        features = task["features"][start:stop].to(device)
        labels = task["labels"][start:stop]
        logits, _ = model.predict_features(features)
        correct += logits.argmax(dim=1).cpu().eq(labels).sum().item()
    return 100.0 * correct / total


def run_experiment(base_config, spec, cache, device):
    kwargs = copy.deepcopy(base_config["classifier"]["kwargs"])
    kwargs.update(
        {
            key: value
            for key, value in spec.items()
            if key
            in {
                "expand_dim",
                "synaptic_degree",
                "coding_level",
                "ridge_lower",
                "ridge_upper",
                "use_projection",
                "use_ridge",
            }
        }
    )
    torch.set_rng_state(cache["projection_rng_state"].clone())
    model = FlyCL(
        IdentityBackbone(kwargs["embedding_dim"]), device, **kwargs
    ).to(device)

    accuracy_matrix = []
    stage_average = []
    ridge_values = []
    post_seconds = []
    started = time.perf_counter()
    for task_idx, train_task in enumerate(cache["train"]):
        model.before_task(task_idx, None, None, None)
        task_started = time.perf_counter()
        model.fit_features(
            train_task["features"].to(device), train_task["labels"].to(device)
        )
        post_seconds.append(time.perf_counter() - task_started)
        ridge_values.append(model.last_ridge)
        row = [
            evaluate_task(model, cache["test"][test_idx], device)
            for test_idx in range(task_idx + 1)
        ]
        accuracy_matrix.append(row)
        stage_average.append(sum(row) / len(row))
        print(
            f"[{spec['name']}] task={task_idx + 1}/10 "
            f"A_t={stage_average[-1]:.4f}"
        )

    final_row = accuracy_matrix[-1]
    bwt = sum(
        final_row[idx] - accuracy_matrix[idx][idx]
        for idx in range(len(final_row) - 1)
    ) / (len(final_row) - 1)
    result = {
        **spec,
        "stage_average_accuracy": stage_average,
        "accuracy_matrix": accuracy_matrix,
        "overall_accuracy": sum(stage_average) / len(stage_average),
        "last_stage_accuracy": stage_average[-1],
        "bwt": bwt,
        "ridge_values": ridge_values,
        "post_seconds_per_task": post_seconds,
        "wall_seconds": time.perf_counter() - started,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def save_results(path, metadata, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "experiments": results}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def validate_complete_results(results):
    """Fail loudly if a supposedly complete sweep is missing or malformed."""
    by_name = {item["name"]: item for item in results}
    if set(by_name) != EXPECTED_EXPERIMENTS:
        missing = EXPECTED_EXPERIMENTS - set(by_name)
        extra = set(by_name) - EXPECTED_EXPERIMENTS
        raise ValueError(f"Incomplete sweep; missing={sorted(missing)}, extra={sorted(extra)}")
    if len(by_name) != len(results):
        raise ValueError("Duplicate experiment names in result file")
    for name, item in by_name.items():
        if len(item["stage_average_accuracy"]) != 10:
            raise ValueError(f"{name}: expected 10 stage accuracies")
        if [len(row) for row in item["accuracy_matrix"]] != list(range(1, 11)):
            raise ValueError(f"{name}: malformed triangular accuracy matrix")
        numeric = [
            item["overall_accuracy"],
            item["last_stage_accuracy"],
            item["bwt"],
            item["wall_seconds"],
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{name}: non-finite metric")

    reference = by_name["full"]["accuracy_matrix"]
    for duplicate in ("m_10000", "p_300", "k_3000"):
        if by_name[duplicate]["accuracy_matrix"] != reference:
            raise ValueError(f"{duplicate} does not reproduce the default full model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fly_cl_cub200.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--scope", choices=("all", "components", "sensitivity"), default="all"
    )
    parser.add_argument(
        "--output", default="outputs/fly_cl_cub200_ablations.json"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional experiment names, for example: --only full wo_ridge m_500",
    )
    parser.add_argument(
        "--rerun",
        nargs="*",
        default=(),
        help="Recompute named experiments even when they exist in the output JSON",
    )
    args = parser.parse_args()

    root = ROOT
    os.chdir(root)
    config = Config(args.config).get_config_dict()
    device = torch.device(f"cuda:{args.device}")
    cache_dir = root / "outputs" / "fly_cl_cub200_feature_cache"
    normalized_cache = load_or_build_cache(
        config,
        True,
        cache_dir / f"normalized_seed{config['seed']}.pt",
        device,
        args.rebuild_cache,
    )
    no_norm_cache = None
    if args.scope in ("all", "components"):
        no_norm_cache = load_or_build_cache(
            config,
            False,
            cache_dir / f"no_normalization_seed{config['seed']}.pt",
            device,
            args.rebuild_cache,
        )

    specs = []
    if args.scope in ("all", "components"):
        specs.extend(COMPONENT_EXPERIMENTS)
    if args.scope in ("all", "sensitivity"):
        specs.extend(SENSITIVITY_EXPERIMENTS)
    if args.only:
        requested = set(args.only)
        known = {spec["name"] for spec in specs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}")
        specs = [spec for spec in specs if spec["name"] in requested]
    rerun = set(args.rerun)
    unknown_reruns = rerun - {spec["name"] for spec in specs}
    if unknown_reruns:
        raise ValueError(f"Unknown rerun experiments: {sorted(unknown_reruns)}")

    output = root / args.output
    previous = {}
    previous_metadata = {}
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            previous = {
                item["name"]: item for item in payload["experiments"]
            }
            previous_metadata = payload.get("metadata", {})
    completed = dict(previous)
    metadata = {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "train_images": sum(len(x["labels"]) for x in normalized_cache["train"]),
        "test_images": sum(len(x["labels"]) for x in normalized_cache["test"]),
        "normalized_feature_seconds": normalized_cache["feature_seconds"],
        "no_normalization_feature_seconds": (
            no_norm_cache["feature_seconds"]
            if no_norm_cache
            else previous_metadata.get("no_normalization_feature_seconds")
        ),
    }
    for spec in specs:
        if spec["name"] in previous and spec["name"] not in rerun:
            print(f"Skipping completed experiment: {spec['name']}")
            continue
        cache = (
            normalized_cache
            if spec.get("normalization", True)
            else no_norm_cache
        )
        print(f"Running experiment: {spec['name']}")
        completed[spec["name"]] = run_experiment(config, spec, cache, device)
        save_results(output, metadata, list(completed.values()))

    complete_results = list(completed.values())
    if set(completed) == EXPECTED_EXPERIMENTS:
        validate_complete_results(complete_results)
    save_results(output, metadata, complete_results)
    print(f"Saved {len(completed)} experiments to {output}")


if __name__ == "__main__":
    main()
