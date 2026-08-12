"""
EfficientNet-B0 Teacher Model for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 wrapper supporting optional feature map extraction.
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.DEFAULT
                base_model = models.efficientnet_b0(weights=weights)
            except (ImportError, AttributeError):
                base_model = models.efficientnet_b0(weights=None)
        else:
            base_model = models.efficientnet_b0(weights=None)

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            base_model.classifier[0],
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)

        if return_features:
            return logits, feat
        return logits


def create_efficientnet_b0(
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """
    Creates an EfficientNet-B0 teacher model for Phyto groundnut leaf classification.
    """
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)


__all__ = ["EfficientNetB0", "create_efficientnet_b0"]
