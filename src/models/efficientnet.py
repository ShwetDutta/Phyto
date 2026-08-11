"""
EfficientNet-B0 Teacher Model for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_efficientnet_b0(
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """
    Creates an EfficientNet-B0 teacher model for Phyto groundnut leaf classification.
    """
    if pretrained:
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except (ImportError, AttributeError):
            model = models.efficientnet_b0(weights=None)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


__all__ = ["create_efficientnet_b0"]
