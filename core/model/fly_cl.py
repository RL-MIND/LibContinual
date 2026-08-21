import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FlyCL(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.device = device

        self.embedding_dim = int(kwargs.get("embedding_dim", backbone.feat_dim))
        self.expand_dim = int(kwargs.get("expand_dim", 10000))
        self.synaptic_degree = int(kwargs.get("synaptic_degree", 300))
        self.coding_level = float(kwargs.get("coding_level", 0.3))
        self.ridge_lower = float(kwargs.get("ridge_lower", 6))
        self.ridge_upper = float(kwargs.get("ridge_upper", 10))
        self.gcv_backend = kwargs.get("gcv_backend", "dual_eigh")
        self.disable_cudnn = bool(kwargs.get("disable_cudnn", False))
        self.use_projection = bool(kwargs.get("use_projection", True))
        self.use_ridge = bool(kwargs.get("use_ridge", True))
        self.total_cls_num = int(kwargs["total_cls_num"])
        self.init_cls_num = int(kwargs["init_cls_num"])
        self.inc_cls_num = int(kwargs["inc_cls_num"])
        self.mask_unseen_classes = bool(kwargs.get("mask_unseen_classes", False))

        if self.embedding_dim != int(backbone.feat_dim):
            raise ValueError(
                f"embedding_dim={self.embedding_dim} does not match "
                f"backbone.feat_dim={backbone.feat_dim}"
            )
        if (
            self.use_projection
            and not 1 <= self.synaptic_degree <= self.embedding_dim
        ):
            raise ValueError("synaptic_degree must be in [1, embedding_dim]")
        if self.use_projection and not 0.0 < self.coding_level <= 1.0:
            raise ValueError("coding_level must be in (0, 1]")
        if self.ridge_upper <= self.ridge_lower:
            raise ValueError("ridge_upper must be greater than ridge_lower")
        if self.gcv_backend not in {"dual_eigh", "svd"}:
            raise ValueError("gcv_backend must be 'dual_eigh' or 'svd'")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        if self.disable_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False

        self.analytic_dim = (
            self.expand_dim if self.use_projection else self.embedding_dim
        )
        projection = self._build_sparse_projection() if self.use_projection else None
        self.register_buffer("projection", projection)
        self.register_buffer("Q", torch.zeros(self.analytic_dim, self.total_cls_num))
        gram = (
            torch.zeros(self.analytic_dim, self.analytic_dim)
            if self.use_ridge
            else None
        )
        self.register_buffer("G", gram)
        self.register_buffer(
            "classifier_weight", torch.zeros(self.analytic_dim, self.total_cls_num)
        )
        self.register_buffer("class_counts", torch.zeros(self.total_cls_num))

        self._optimizer_anchor = nn.Parameter(torch.zeros(()))

        self._classes_seen_so_far = 0
        self._is_fitted = False
        self.last_ridge = None
        self.last_feature_time = 0.0
        self.last_post_time = 0.0

    def _build_sparse_projection(self):
        """Create the reference implementation's sparse Gaussian projection."""
        projection = torch.zeros(self.expand_dim, self.embedding_dim)
        for row in range(self.expand_dim):
            selected = torch.randperm(self.embedding_dim)[: self.synaptic_degree]
            projection[row, selected] = torch.randn(self.synaptic_degree)
        return projection.to_sparse_csc()

    def train(self, mode=True):
        # Trainer calls model.train(); the pre-trained feature extractor must
        # nevertheless stay deterministic and frozen, as in the official code.
        super().train(mode)
        self.backbone.eval()
        return self

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        increment = self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self._classes_seen_so_far += increment
        if self._classes_seen_so_far > self.total_cls_num:
            raise ValueError("Seen classes exceed total_cls_num")
        print(
            f"[FlyCL] Task {task_idx}: frozen feature extraction; "
            "analytic classifier will be updated after the task."
        )

    def observe(self, data):
        loss = self._optimizer_anchor * 0.0
        return None, 0.0, loss

    @torch.no_grad()
    def _extract_features(self, data_loader):
        features = []
        labels = []
        self.backbone.eval()
        for batch in tqdm(data_loader, desc="Fly-CL feature extraction"):
            images = batch["image"].to(self.device, non_blocking=True)
            feature = self.backbone(images)
            features.append(feature)
            labels.append(batch["label"].to(self.device, non_blocking=True))
        if not features:
            raise ValueError("Cannot fit Fly-CL on an empty dataloader")
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def _fly_encode(self, features):
        if not self.use_projection:
            return features.T
        expanded = torch.sparse.mm(self.projection, features.T)
        topk = max(1, int(self.expand_dim * self.coding_level))
        values, indices = expanded.topk(topk, dim=0, largest=True)
        encoded = torch.zeros_like(expanded)
        encoded.scatter_(0, indices, values)
        return encoded

    @torch.no_grad()
    def _select_ridge_parameter(self, features, targets):
        """Generalized cross-validation used by the official implementation."""
        if self.gcv_backend == "svd":
            u, singular_values, _ = torch.linalg.svd(features, full_matrices=False)
            singular_sq = singular_values.square()
        else:
            sample_gram = features @ features.T
            singular_sq, u = torch.linalg.eigh(sample_gram)
            singular_sq.clamp_(min=0.0)
        uty = u.T @ targets
        exponents = np.arange(self.ridge_lower, self.ridge_upper)
        ridges = torch.as_tensor(
            10.0 ** exponents,
            dtype=features.dtype,
            device=features.device,
        )
        n_samples = features.shape[0]
        scores = []
        for ridge in ridges:
            diagonal = singular_sq / (singular_sq + ridge)
            degrees_of_freedom = diagonal.sum()
            prediction = u @ (diagonal[:, None] * uty)
            residual = torch.linalg.vector_norm(targets - prediction).square()
            denominator = (1.0 - degrees_of_freedom / n_samples).square()
            scores.append((residual / n_samples) / denominator)
        return ridges[torch.argmin(torch.stack(scores))]

    @torch.no_grad()
    def fit_features(self, features, labels):
        """Update the analytic head from already-extracted backbone features."""
        encoded = self._fly_encode(features)
        del features
        targets = F.one_hot(labels.long(), self.total_cls_num).to(encoded.dtype)
        self.class_counts.add_(targets.sum(dim=0))
        del labels
        self.Q.add_(encoded @ targets)
        if self.use_ridge:
            self.G.add_(encoded @ encoded.T)
            ridge = self._select_ridge_parameter(encoded.T, targets)
            del encoded, targets
            regularized = self.G.clone()
            regularized.diagonal().add_(ridge)
            cholesky = torch.linalg.cholesky(regularized)
            del regularized
            self.classifier_weight.copy_(torch.cholesky_solve(self.Q, cholesky))
            self.last_ridge = float(ridge.item())
        else:
            del encoded, targets
            counts = self.class_counts.clamp_min(1.0).unsqueeze(0)
            self.classifier_weight.copy_(self.Q / counts)
            self.last_ridge = None

        self._is_fitted = True

    @torch.no_grad()
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        total_start = time.perf_counter()
        feature_start = time.perf_counter()
        features, labels = self._extract_features(train_loader)
        self.last_feature_time = time.perf_counter() - feature_start
        self.fit_features(features, labels)
        self.last_post_time = (
            time.perf_counter() - total_start - self.last_feature_time
        )
        print(
            f"[FlyCL] Task {task_idx} fitted: "
            f"ridge={self.last_ridge if self.last_ridge is not None else 'disabled'}, "
            f"feature_time={self.last_feature_time:.2f}s, "
            f"post_time={self.last_post_time:.2f}s"
        )

    @torch.no_grad()
    def predict_features(self, features, targets=None):
        """Predict from cached backbone features using the fitted analytic head."""
        if not self._is_fitted:
            raise RuntimeError("Fly-CL classifier is not fitted; call after_task first")
        encoded = self._fly_encode(features)
        if self.use_ridge:
            if self.use_projection:
                logits = torch.sparse.mm(
                    encoded.T.to_sparse_csc(), self.classifier_weight
                )
            else:
                logits = encoded.T @ self.classifier_weight
        else:
            encoded = F.normalize(encoded, dim=0)
            prototypes = F.normalize(self.classifier_weight, dim=0)
            if self.use_projection:
                logits = torch.sparse.mm(encoded.T.to_sparse_csc(), prototypes)
            else:
                logits = encoded.T @ prototypes
        if self.mask_unseen_classes:
            logits = logits[:, : self._classes_seen_so_far]

        accuracy = None
        if targets is not None:
            predictions = logits.argmax(dim=1).cpu()
            accuracy = predictions.eq(targets.cpu()).float().mean().item()
        return logits, accuracy

    @torch.no_grad()
    def inference(self, data):
        images = data["image"].to(self.device, non_blocking=True)
        features = self.backbone(images)
        return self.predict_features(features, data["label"])

    def get_parameters(self, config):
        return [self._optimizer_anchor]
