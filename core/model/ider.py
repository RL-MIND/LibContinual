import copy
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .finetune import Finetune


class IDER(Finetune):
    """Idempotent Experience Replay for LibContinual.

    This follows the official implementation: the current batch receives the
    standard idempotent CE loss, while replay samples are drawn inside
    ``observe`` for supervised ER and old-checkpoint idempotent distillation.
    """

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.classifier = nn.Identity()
        self.weighta = kwargs.get("weighta", 0.5)
        self.weightb = kwargs.get("weightb", 0.4)
        self.weightc = kwargs.get("weightc", 0.5)
        self.empty_prob = kwargs.get("weightmask", kwargs.get("empty_prob", 1.0))
        self.refine_inference = kwargs.get("refine_inference", False)
        self.mask_unseen = kwargs.get("mask_unseen", False)
        self.old_backbone = None
        self.old_seen_cls_num = 0
        self.seen_cls_num = num_class
        self.task_idx = 0
        self.buffer = None
        self.buffer_dataset = None
        self.minibatch_size = kwargs.get("minibatch_size", kwargs.get("buffer_batch_size", 32))
        self.online_buffer_update = kwargs.get("online_buffer_update", True)
        self.class_balance = kwargs.get("class_balance", True)
        self.print_buffer_stats = kwargs.get("print_buffer_stats", False)
        self.buffer_update_after_step = kwargs.get("buffer_update_after_step", False)
        self.buffer_storage = kwargs.get("buffer_storage", "raw")
        self.ce_detach_signal = kwargs.get("ce_detach_signal", True)
        self.distill_detach_signal = kwargs.get("distill_detach_signal", True)
        self.old_model_eval = kwargs.get("old_model_eval", True)
        self.freeze_old_model = kwargs.get("freeze_old_model", True)
        self._pending_buffer_data = None
        self._pending_buffer_labels = None
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

    def _empty_signal(self, batch_size, upto=None):
        upto = self.num_class if upto is None else upto
        if hasattr(self.backbone, "empty_signal"):
            return self.backbone.empty_signal(batch_size, self.device, upto=upto)
        signal = torch.zeros(batch_size, self.num_class, device=self.device)
        signal[:, :upto] = 1.0 / max(upto, 1)
        return signal

    def _one_hot(self, labels):
        return F.one_hot(labels, num_classes=self.num_class).float()

    def _sample_second_input(self, labels):
        empty = self._empty_signal(labels.size(0))
        one_hot = self._one_hot(labels)
        if self.empty_prob >= 1:
            return empty
        if self.empty_prob <= 0:
            return one_hot
        use_empty = (torch.rand(1, device=self.device) <= self.empty_prob).item()
        return empty if use_empty else one_hot

    def _mask_logits(self, logits, upto=None):
        if not self.mask_unseen:
            return logits
        upto = self.seen_cls_num if upto is None else upto
        if upto >= logits.size(1):
            return logits
        masked = logits.clone()
        masked[:, upto:] = -1e9
        return masked

    def _backbone_logits(self, x, y_signal=None, upto=None, backbone=None):
        backbone = self.backbone if backbone is None else backbone
        try:
            output = backbone(x, y_signal, upto=upto)
        except TypeError:
            output = backbone(x, y_signal)

        if isinstance(output, dict):
            return output["logits"]
        if isinstance(output, tuple):
            return output[0]
        return output

    def _refined_logits(self, x):
        empty = self._empty_signal(x.size(0))
        logits0 = self._backbone_logits(x, empty)
        logits0 = self._mask_logits(logits0)
        if not self.refine_inference:
            return logits0

        probs0 = F.softmax(logits0, dim=1).detach()
        logits1 = self._backbone_logits(x, probs0)
        return self._mask_logits(logits1)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.seen_cls_num = self.kwargs["init_cls_num"] + task_idx * self.kwargs["inc_cls_num"]
        self.seen_cls_num = min(self.seen_cls_num, self.num_class)
        self.buffer = buffer
        self.buffer_dataset = train_loader.dataset if train_loader is not None else None
        if buffer is not None and hasattr(buffer, "batch_size"):
            self.minibatch_size = buffer.batch_size
        if self.old_backbone is not None:
            self.old_backbone = self.old_backbone.to(self.device)
            if self.old_model_eval:
                self.old_backbone.eval()
            else:
                self.old_backbone.train()

    def _load_buffer_image(self, image_ref):
        dataset = self.buffer_dataset
        if dataset is None:
            raise RuntimeError("IDER needs the train dataset to sample replay images.")

        if torch.is_tensor(image_ref):
            image = self._to_pil(image_ref.cpu())
        elif dataset.dataset in ["binary_cifar10", "binary_cifar100"]:
            image = Image.fromarray(np.uint8(image_ref))
        elif dataset.dataset == "tiny-imagenet":
            image = Image.open(image_ref).convert("RGB")
        elif dataset.dataset == "processed_tinyimg":
            if np.max(image_ref) <= 1.0:
                image_ref = np.uint8(255 * image_ref)
            image = Image.fromarray(np.uint8(image_ref)).convert("RGB")
        else:
            image = Image.open(os.path.join(dataset.data_root, dataset.mode, image_ref)).convert("RGB")
        return dataset.trfms(image)

    def _make_buffer_image_ref(self, image_ref):
        if self.buffer_storage != "tensor":
            return image_ref

        if torch.is_tensor(image_ref):
            return image_ref.detach().cpu()

        dataset = self.buffer_dataset
        if dataset.dataset in ["binary_cifar10", "binary_cifar100"]:
            image = Image.fromarray(np.uint8(image_ref))
        elif dataset.dataset == "tiny-imagenet":
            image = Image.open(image_ref).convert("RGB")
        elif dataset.dataset == "processed_tinyimg":
            if np.max(image_ref) <= 1.0:
                image_ref = np.uint8(255 * image_ref)
            image = Image.fromarray(np.uint8(image_ref)).convert("RGB")
        else:
            image = Image.open(os.path.join(dataset.data_root, dataset.mode, image_ref)).convert("RGB")
        return self._to_tensor(image).cpu()

    def _sample_buffer_batch(self):
        if self.buffer is None or self.buffer.is_empty():
            return None

        n_items = len(self.buffer.labels)
        batch_size = min(self.minibatch_size, n_items)
        indices = np.random.choice(n_items, size=batch_size, replace=False)
        images = [self._load_buffer_image(self.buffer.images[idx]) for idx in indices]
        labels = [self.buffer.labels[idx] for idx in indices]
        return torch.stack(images).to(self.device), torch.tensor(labels, dtype=torch.long, device=self.device)

    def _online_update_buffer(self, data, labels):
        if (
            not self.online_buffer_update
            or self.buffer is None
            or self.buffer.buffer_size <= 0
            or self.buffer_dataset is None
            or "index" not in data
        ):
            return

        if not hasattr(self.buffer, "num_seen_examples"):
            self.buffer.num_seen_examples = len(self.buffer.labels)

        indices = data["index"].detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()

        for dataset_idx, label in zip(indices, labels):
            image_ref = self._make_buffer_image_ref(self.buffer_dataset.images[int(dataset_idx)])
            seen = self.buffer.num_seen_examples
            if len(self.buffer.labels) < self.buffer.buffer_size:
                self.buffer.images.append(image_ref)
                self.buffer.labels.append(int(label))
            else:
                replace_idx = np.random.randint(0, seen + 1)
                if replace_idx < self.buffer.buffer_size:
                    if self.class_balance and len(self.buffer.labels) > 0:
                        counts = Counter(self.buffer.labels)
                        max_count = max(counts.values())
                        majority_classes = {cls for cls, cnt in counts.items() if cnt == max_count}
                        majority_indices = [
                            idx for idx, old_label in enumerate(self.buffer.labels)
                            if old_label in majority_classes
                        ]
                        replace_idx = int(np.random.choice(majority_indices))
                    self.buffer.images[replace_idx] = image_ref
                    self.buffer.labels[replace_idx] = int(label)
            self.buffer.num_seen_examples += 1

    def _idempotent_ce_loss(self, x, y):
        y_signal = self._sample_second_input(y)
        logits1 = self._backbone_logits(x, y_signal)
        signal_logits = logits1.detach() if self.ce_detach_signal else logits1
        logits2 = self._backbone_logits(x, F.softmax(signal_logits, dim=1))
        loss1 = self.loss_fn(self._mask_logits(logits1), y)
        loss2 = self.loss_fn(self._mask_logits(logits2), y)
        return 0.5 * (loss1 + loss2), logits1, logits2, y_signal

    def _distill_loss(self, x):
        uniform = self._empty_signal(x.size(0))
        logits1 = self._backbone_logits(x, uniform)
        signal_logits = logits1.detach() if self.distill_detach_signal else logits1
        if self.old_model_eval:
            self.old_backbone.eval()
        else:
            self.old_backbone.train()
        logits2 = self._backbone_logits(
            x,
            F.softmax(signal_logits, dim=1),
            backbone=self.old_backbone,
        )
        return F.mse_loss(logits1, logits2)

    def observe(self, data):
        x, y = data["image"], data["label"]
        x = x.to(self.device)
        y = y.to(self.device)

        loss, logits1, logits2, _ = self._idempotent_ce_loss(x, y)

        if self.old_backbone is not None and self.old_seen_cls_num > 0 and self.weightb > 0:
            loss = loss + self.weightb * self._distill_loss(x)

        if self.weightc != 0:
            buffer_batch = self._sample_buffer_batch()
            if buffer_batch is not None:
                buffer_loss, _, _, _ = self._idempotent_ce_loss(*buffer_batch)
                loss = loss + self.weightc * 2.0 * buffer_loss

        if self.old_backbone is not None and self.old_seen_cls_num > 0 and self.weighta != 0:
            buffer_batch = self._sample_buffer_batch()
            if buffer_batch is not None:
                loss = loss + self.weighta * self._distill_loss(buffer_batch[0])

        pred = torch.argmax(self._mask_logits(logits2), dim=1)
        acc = torch.sum(pred == y).item()
        if self.buffer_update_after_step:
            self._pending_buffer_data = data
            self._pending_buffer_labels = y.detach()
        else:
            self._online_update_buffer(data, y)
        return pred, acc / x.size(0), loss

    def after_observe(self):
        if self._pending_buffer_data is None or self._pending_buffer_labels is None:
            return
        self._online_update_buffer(self._pending_buffer_data, self._pending_buffer_labels)
        self._pending_buffer_data = None
        self._pending_buffer_labels = None

    def _task_class_range(self, task_id):
        if task_id < 0:
            return 0, self.num_class
        if task_id == 0:
            start = 0
            end = self.kwargs["init_cls_num"]
        else:
            start = self.kwargs["init_cls_num"] + (task_id - 1) * self.kwargs["inc_cls_num"]
            end = start + self.kwargs["inc_cls_num"]
        return start, min(end, self.num_class)

    def inference(self, data, task_id=-1):
        x, y = data["image"], data["label"]
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self._refined_logits(x)
        if task_id > -1:
            start, end = self._task_class_range(task_id)
            task_logits = torch.full_like(logits, -1e9)
            task_logits[:, start:end] = logits[:, start:end]
            logits = task_logits
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def predict_logits(self, data):
        x = data["image"].to(self.device)
        return self._refined_logits(x)

    def forward(self, x):
        return self._refined_logits(x)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.print_buffer_stats and buffer is not None and hasattr(buffer, "labels") and len(buffer.labels) > 0:
            counts = Counter(buffer.labels)
            seen_classes = self.kwargs["init_cls_num"] + task_idx * self.kwargs["inc_cls_num"]
            seen_classes = min(seen_classes, self.num_class)
            values = [counts.get(cls, 0) for cls in range(seen_classes)]
            zero_classes = sum(1 for value in values if value == 0)
            print(
                "[IDER] Buffer class counts after task "
                f"{task_idx}: size={len(buffer.labels)}, classes={len(counts)}, "
                f"min={min(values)}, max={max(values)}, zero={zero_classes}, "
                f"head={values[:min(20, len(values))]}"
            )
        self.old_backbone = copy.deepcopy(self.backbone).to(self.device)
        if self.old_model_eval:
            self.old_backbone.eval()
        else:
            self.old_backbone.train()
        for param in self.old_backbone.parameters():
            param.requires_grad = not self.freeze_old_model
        self.old_seen_cls_num = self.seen_cls_num

    def get_parameters(self, config):
        return [{"params": self.backbone.parameters()}]
