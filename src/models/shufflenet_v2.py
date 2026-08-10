"""
ShuffleNetV2 Baseline and CBAM-Enhanced Model Architectures for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.cbam import CBAM


class ShuffleNetV2CBAM(nn.Module):
    """
    ShuffleNetV2 x1.0 with integrated CBAM attention blocks at stages 2, 3, 4, and conv5.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if pretrained:
            try:
                from torchvision.models import ShuffleNet_V2_X1_0_Weights
                weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
                base_model = models.shufflenet_v2_x1_0(weights=weights)
            except (ImportError, AttributeError):
                base_model = models.shufflenet_v2_x1_0(weights=None)
        else:
            base_model = models.shufflenet_v2_x1_0(weights=None)

        self.conv1 = base_model.conv1
        self.maxpool = base_model.maxpool

        # Stage 2 (116 channels)
        self.stage2 = base_model.stage2
        self.cbam2 = CBAM(116, reduction=reduction, kernel_size=kernel_size)

        # Stage 3 (232 channels)
        self.stage3 = base_model.stage3
        self.cbam3 = CBAM(232, reduction=reduction, kernel_size=kernel_size)

        # Stage 4 (464 channels)
        self.stage4 = base_model.stage4
        self.cbam4 = CBAM(464, reduction=reduction, kernel_size=kernel_size)

        # Conv5 (1024 channels)
        self.conv5 = base_model.conv5
        self.cbam5 = CBAM(1024, reduction=reduction, kernel_size=kernel_size)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.cbam2(x)

        x = self.stage3(x)
        x = self.cbam3(x)

        x = self.stage4(x)
        x = self.cbam4(x)

        x = self.conv5(x)
        x = self.cbam5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_shufflenet_v2(
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """
    Creates a baseline ShuffleNetV2 x1.0 model with a custom classifier head.
    """
    if pretrained:
        try:
            from torchvision.models import ShuffleNet_V2_X1_0_Weights
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
            model = models.shufflenet_v2_x1_0(weights=weights)
        except (ImportError, AttributeError):
            model = models.shufflenet_v2_x1_0(weights=None)
    else:
        model = models.shufflenet_v2_x1_0(weights=None)

    model.fc = nn.Linear(1024, num_classes)
    return model


def create_shufflenet_v2_cbam(
    num_classes: int = 6,
    pretrained: bool = True,
    reduction: int = 16,
    kernel_size: int = 7,
) -> nn.Module:
    """
    Creates a CBAM-enhanced ShuffleNetV2 x1.0 model with a custom classifier head.
    """
    return ShuffleNetV2CBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        reduction=reduction,
        kernel_size=kernel_size,
    )


def create_shufflenet_v2_x0_5(
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """
    Creates an ultra-lightweight ShuffleNetV2 x0.5 model for Edge-AI Knowledge Distillation.
    """
    if pretrained:
        try:
            from torchvision.models import ShuffleNet_V2_X0_5_Weights
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
            model = models.shufflenet_v2_x0_5(weights=weights)
        except (ImportError, AttributeError):
            model = models.shufflenet_v2_x0_5(weights=None)
    else:
        model = models.shufflenet_v2_x0_5(weights=None)

    model.fc = nn.Linear(1024, num_classes)
    return model
