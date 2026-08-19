import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["resnet18_id2"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, temperature=12.0):
        super().__init__()
        fc = nn.Linear(feat_dim, num_classes)
        self.weight = nn.Parameter(fc.weight.t())
        self.bias = nn.Parameter(fc.bias)
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)

    def forward(self, features):
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
        weight = F.normalize(self.weight, p=2, dim=0, eps=1e-12)
        return self.temperature * torch.mm(features, weight)


class ResNetStage1(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super().__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        return self.layer2(out)


class ResNetStage2(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, use_cos=False):
        super().__init__()
        self.in_planes = nf * 2
        self.num_classes = num_classes
        self.out_dim = nf * 8 * block.expansion
        self.label_fc = nn.Sequential(
            nn.Linear(num_classes, nf * 2),
            nn.LeakyReLU(inplace=True),
        )
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        if use_cos:
            self.classifier = CosineClassifier(self.out_dim, num_classes)
        else:
            self.classifier = nn.Linear(self.out_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, features, label_signal):
        out = features + self.label_fc(label_signal)[..., None, None]
        out = self.layer3(out)
        out = self.layer4(out)
        pooled = F.avg_pool2d(out, out.shape[2])
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits[:, : self.num_classes], flat, out


class IdempotentResNet(nn.Module):
    """ResNet-18 split for IDER.

    The image path produces an intermediate feature map. The second input is a
    label/probability signal projected to the same channel dimension and added
    before the later ResNet stages, following the official IDER implementation.
    """

    def __init__(self, block, num_blocks, num_classes=100, nf=64, use_cos=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = nf * 8 * block.expansion
        self.f1 = ResNetStage1(block, num_blocks, nf)
        self.f2 = ResNetStage2(block, num_blocks, num_classes, nf, use_cos=use_cos)

    def empty_signal(self, batch_size, device, upto=None):
        upto = self.num_classes if upto is None else min(upto, self.num_classes)
        signal = torch.zeros(batch_size, self.num_classes, device=device)
        signal[:, :upto] = 1.0 / max(upto, 1)
        return signal

    def forward(self, x, y=None, upto=None, returnt="all"):
        if y is None:
            y = self.empty_signal(x.size(0), x.device, upto=upto)

        stage1_features = self.f1(x)
        logits, features, fmap = self.f2(stage1_features, y)

        if returnt == "logits":
            return logits
        if returnt == "features":
            return {"features": features}
        return {"logits": logits, "features": features, "fmaps": [stage1_features, fmap]}


def resnet18_id2(pretrained=False, progress=True, num_classes=100, nf=64, use_cos=False, **kwargs):
    if "nclasses" in kwargs:
        num_classes = kwargs["nclasses"]
    return IdempotentResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        nf=nf,
        use_cos=use_cos,
    )
