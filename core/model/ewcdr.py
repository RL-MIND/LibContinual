import torch
import torch.nn as nn
import torch.nn.functional as F

from .finetune import Finetune


class IncrementalLinearNet(nn.Module):
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_class)

    def _features(self, x):
        output = self.backbone(x)
        if isinstance(output, dict):
            return output["features"]
        return output

    def forward(self, x):
        return self.classifier(self._features(x))

    def update_classifier(self, num_class):
        old_classifier = self.classifier
        if old_classifier.out_features == num_class:
            return

        new_classifier = nn.Linear(old_classifier.in_features, num_class)
        new_classifier = new_classifier.to(old_classifier.weight.device)
        with torch.no_grad():
            old_out = old_classifier.out_features
            new_classifier.weight[:old_out].copy_(old_classifier.weight)
            new_classifier.bias[:old_out].copy_(old_classifier.bias)
        self.classifier = new_classifier


class EWCDR(Finetune):
    """EWC Done Right with logits reversal during importance estimation."""

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.lamda = kwargs["lamda"]
        self.omega_max = kwargs.get("omega_max", 1e-4)

        self.task_idx = 0
        self.known_cls_num = 0
        self.total_cls_num = self.init_cls_num
        self.network = IncrementalLinearNet(backbone, feat_dim, self.init_cls_num)
        self.ref_param = {}
        self.omega = None

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.known_cls_num = (
            0 if task_idx == 0 else self.init_cls_num + (task_idx - 1) * self.inc_cls_num
        )
        self.total_cls_num = self.init_cls_num + task_idx * self.inc_cls_num
        self.network.update_classifier(self.total_cls_num)
        self.network.to(self.device)

    def observe(self, data):
        x = data["image"].to(self.device)
        y = data["label"].to(self.device)
        logits = self.network(x)

        if self.task_idx == 0:
            loss = F.cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(
                logits[:, self.known_cls_num:], y - self.known_cls_num
            )
            loss = loss + self.lamda * self.compute_ewc()

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def inference(self, data):
        x = data["image"].to(self.device)
        y = data["label"].to(self.device)
        logits = self.network(x)
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        new_omega = self.get_importance(train_loader)
        if self.omega is not None:
            alpha = self.known_cls_num / self.total_cls_num
            for name, old_omega in self.omega.items():
                if name not in new_omega:
                    continue
                old_slice = self._prefix_slice(old_omega)
                new_omega[name][old_slice] = (
                    alpha * old_omega + (1 - alpha) * new_omega[name][old_slice]
                )

        self.omega = new_omega
        self.ref_param = {
            name: param.clone().detach()
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }

    def get_importance(self, train_loader):
        omega = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }

        was_training = self.network.training
        self.network.train()
        for data in train_loader:
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            logits = -self.network(x)
            loss = F.cross_entropy(logits, y)

            self.network.zero_grad(set_to_none=True)
            loss.backward()

            for name, param in self.network.named_parameters():
                if param.grad is not None and name in omega:
                    omega[name] += param.grad.pow(2).detach()

        for name, value in omega.items():
            value = value / len(train_loader)
            omega[name] = torch.clamp(value, max=self.omega_max)

        self.network.zero_grad(set_to_none=True)
        self.network.train(was_training)
        return omega

    def compute_ewc(self):
        if self.omega is None:
            return torch.zeros((), device=self.device)

        loss = torch.zeros((), device=self.device)
        for name, param in self.network.named_parameters():
            if name not in self.omega:
                continue
            ref = self.ref_param[name]
            old_slice = self._prefix_slice(ref)
            loss += torch.sum(self.omega[name] * (param[old_slice] - ref).pow(2)) / 2
        return loss

    @staticmethod
    def _prefix_slice(tensor):
        return (slice(0, tensor.shape[0]),) + (slice(None),) * (tensor.dim() - 1)

    def forward(self, x):
        return self.network(x)

    def get_parameters(self, config):
        return [{"params": self.network.parameters()}]
