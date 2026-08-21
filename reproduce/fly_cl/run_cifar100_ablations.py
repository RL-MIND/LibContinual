"""Run the five Fly-CL component ablations on CIFAR-100.

The script caches the frozen ViT features once for each input-normalization
setting and delegates every analytic update and prediction to ``FlyCL``.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from reproduce.fly_cl.run_cub200_ablations import (
    COMPONENT_EXPERIMENTS,
    load_or_build_cache,
    run_experiment,
    save_results,
)


EXPECTED_EXPERIMENTS = {spec["name"] for spec in COMPONENT_EXPERIMENTS}


def validate_complete_results(results):
    by_name = {item["name"]: item for item in results}
    if set(by_name) != EXPECTED_EXPERIMENTS:
        missing = EXPECTED_EXPERIMENTS - set(by_name)
        extra = set(by_name) - EXPECTED_EXPERIMENTS
        raise ValueError(
            f"Incomplete CIFAR sweep; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if len(by_name) != len(results):
        raise ValueError("Duplicate CIFAR experiment names")
    for name, item in by_name.items():
        if len(item["stage_average_accuracy"]) != 10:
            raise ValueError(f"{name}: expected 10 stage accuracies")
        if [len(row) for row in item["accuracy_matrix"]] != list(range(1, 11)):
            raise ValueError(f"{name}: malformed triangular accuracy matrix")
        metrics = (
            item["overall_accuracy"],
            item["last_stage_accuracy"],
            item["bwt"],
            item["wall_seconds"],
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError(f"{name}: non-finite metric")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fly_cl_cifar100.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--output", default="outputs/fly_cl_cifar100_ablations.json"
    )
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--rerun", nargs="*", default=())
    args = parser.parse_args()

    os.chdir(ROOT)
    config = Config(args.config).get_config_dict()
    device = torch.device(f"cuda:{args.device}")
    cache_dir = ROOT / "outputs" / "fly_cl_cifar100_feature_cache"
    normalized_cache = load_or_build_cache(
        config,
        True,
        cache_dir / f"normalized_seed{config['seed']}.pt",
        device,
        args.rebuild_cache,
    )
    no_norm_cache = load_or_build_cache(
        config,
        False,
        cache_dir / f"no_normalization_seed{config['seed']}.pt",
        device,
        args.rebuild_cache,
    )

    specs = list(COMPONENT_EXPERIMENTS)
    if args.only:
        requested = set(args.only)
        unknown = requested - EXPECTED_EXPERIMENTS
        if unknown:
            raise ValueError(f"Unknown CIFAR experiments: {sorted(unknown)}")
        specs = [spec for spec in specs if spec["name"] in requested]
    rerun = set(args.rerun)
    unknown_reruns = rerun - {spec["name"] for spec in specs}
    if unknown_reruns:
        raise ValueError(f"Unknown CIFAR reruns: {sorted(unknown_reruns)}")

    output = ROOT / args.output
    previous = {}
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            previous = {
                item["name"]: item for item in json.load(handle)["experiments"]
            }
    completed = dict(previous)
    metadata = {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "train_images": sum(len(x["labels"]) for x in normalized_cache["train"]),
        "test_images": sum(len(x["labels"]) for x in normalized_cache["test"]),
        "normalized_feature_seconds": normalized_cache["feature_seconds"],
        "no_normalization_feature_seconds": no_norm_cache["feature_seconds"],
    }

    for spec in specs:
        if spec["name"] in previous and spec["name"] not in rerun:
            print(f"Skipping completed experiment: {spec['name']}")
            continue
        cache = normalized_cache if spec.get("normalization", True) else no_norm_cache
        print(f"Running CIFAR-100 experiment: {spec['name']}")
        completed[spec["name"]] = run_experiment(config, spec, cache, device)
        save_results(output, metadata, list(completed.values()))

    results = list(completed.values())
    if set(completed) == EXPECTED_EXPERIMENTS:
        validate_complete_results(results)
    save_results(output, metadata, results)
    print(f"Saved {len(results)} CIFAR-100 experiments to {output}")


if __name__ == "__main__":
    main()
