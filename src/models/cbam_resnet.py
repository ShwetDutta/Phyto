"""
ResNet50 with integrated Convolutional Block Attention Module (CBAM) for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.cbam import CBAM


class ResNet50CBAM(nn.Module):
    """
    ResNet50 with CBAM attention blocks integrated after each of the 4 residual stages
    (layer1: 256 channels, layer2: 512 channels, layer3: 1024 channels, layer4: 2048 channels).
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
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
                base_model = models.resnet50(weights=weights)
            except (ImportError, AttributeError):
                base_model = models.resnet50(weights=None)
        else:
            base_model = models.resnet50(weights=None)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Stage 1 (256 channels)
        self.layer1 = base_model.layer1
        self.cbam1 = CBAM(256, reduction=reduction, kernel_size=kernel_size)

        # Stage 2 (512 channels)
        self.layer2 = base_model.layer2
        self.cbam2 = CBAM(512, reduction=reduction, kernel_size=kernel_size)

        # Stage 3 (1024 channels)
        self.layer3 = base_model.layer3
        self.cbam3 = CBAM(1024, reduction=reduction, kernel_size=kernel_size)

        # Stage 4 (2048 channels)
        self.layer4 = base_model.layer4
        self.cbam4 = CBAM(2048, reduction=reduction, kernel_size=kernel_size)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet50_cbam(
    num_classes: int = 6,
    pretrained: bool = True,
    reduction: int = 16,
    kernel_size: int = 7,
) -> nn.Module:
    """
    Creates a CBAM-enhanced ResNet50 teacher model for Phyto groundnut leaf classification.
    """
    return ResNet50CBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        reduction=reduction,
        kernel_size=kernel_size,
    )


__all__ = ["ResNet50CBAM", "create_resnet50_cbam"]
