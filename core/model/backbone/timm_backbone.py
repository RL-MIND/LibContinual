import timm
import torch.nn as nn


class TimmBackbone(nn.Module):

    def __init__(self, model_name="vit_base_patch16_224", pretrained=False, **kwargs):
        super().__init__()
        self.feat = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )
        self.feat_dim = self.feat.num_features

    def forward(self, x):
        return self.feat(x)


def timm_backbone(pretrained=False, **kwargs):
    return TimmBackbone(pretrained=pretrained, **kwargs)
