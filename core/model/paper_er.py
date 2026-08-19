import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .finetune import Finetune


class PaperER(Finetune):
    """Online ER baseline used by the original IDER code.

    Each training step optimizes CE on the current mini-batch concatenated with
    a random replay mini-batch. The current mini-batch is inserted into the
    replay memory by reservoir sampling after the optimizer step.
    """

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.buffer = None
        self.buffer_dataset = None
        self.minibatch_size = kwargs.get("minibatch_size", kwargs.get("buffer_batch_size", 32))
        self.buffer_storage = kwargs.get("buffer_storage", "tensor")
        self._pending_buffer_data = None
        self._pending_buffer_labels = None
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.buffer = buffer
        self.buffer_dataset = train_loader.dataset if train_loader is not None else None
        if buffer is not None and hasattr(buffer, "batch_size"):
            self.minibatch_size = buffer.batch_size

    def _logits(self, x):
        return self.classifier(self.backbone(x)["features"])

    def _load_buffer_image(self, image_ref):
        dataset = self.buffer_dataset
        if dataset is None:
            raise RuntimeError("PaperER needs the train dataset to sample replay images.")

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

        dataset = self.buffer_dataset
        if torch.is_tensor(image_ref):
            return image_ref.detach().cpu()
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

        batch_size = min(self.minibatch_size, len(self.buffer.labels))
        indices = np.random.choice(len(self.buffer.labels), size=batch_size, replace=False)
        images = [self._load_buffer_image(self.buffer.images[idx]) for idx in indices]
        labels = [self.buffer.labels[idx] for idx in indices]
        return torch.stack(images).to(self.device), torch.tensor(labels, dtype=torch.long, device=self.device)

    def _online_update_buffer(self, data, labels):
        if (
            self.buffer is None
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
                    self.buffer.images[replace_idx] = image_ref
                    self.buffer.labels[replace_idx] = int(label)
            self.buffer.num_seen_examples += 1

    def observe(self, data):
        x, y = data["image"].to(self.device), data["label"].to(self.device)
        real_batch_size = x.size(0)

        replay_batch = self._sample_buffer_batch()
        if replay_batch is not None:
            buf_x, buf_y = replay_batch
            train_x = torch.cat((x, buf_x), dim=0)
            train_y = torch.cat((y, buf_y), dim=0)
        else:
            train_x, train_y = x, y

        logits = self._logits(train_x)
        loss = self.loss_fn(logits, train_y)

        current_logits = logits[:real_batch_size]
        pred = torch.argmax(current_logits, dim=1)
        acc = torch.sum(pred == y).item()

        self._pending_buffer_data = data
        self._pending_buffer_labels = y.detach()
        return pred, acc / real_batch_size, loss

    def after_observe(self):
        if self._pending_buffer_data is None or self._pending_buffer_labels is None:
            return
        self._online_update_buffer(self._pending_buffer_data, self._pending_buffer_labels)
        self._pending_buffer_data = None
        self._pending_buffer_labels = None

    def inference(self, data, task_id=-1):
        x, y = data["image"].to(self.device), data["label"].to(self.device)
        logits = self._logits(x)
        if task_id > -1:
            start, end = self._task_class_range(task_id)
            task_logits = torch.full_like(logits, -1e9)
            task_logits[:, start:end] = logits[:, start:end]
            logits = task_logits
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

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

    def predict_logits(self, data):
        x = data["image"].to(self.device)
        return self._logits(x)
