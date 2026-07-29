import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from phyto.models.cbam import CBAM

class BaselineShuffleNetV2(nn.Module):
    """
    Baseline Lightweight ShuffleNetV2 Model (1.0x width multiplier).
    Standard architecture without Attention Modules.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(BaselineShuffleNetV2, self).__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        base_model = shufflenet_v2_x1_0(weights=weights)

        self.conv1 = base_model.conv1
        self.maxpool = base_model.maxpool
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base_model.stage4
        self.conv5 = base_model.conv5

        # Replace classification head
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # Global Average Pooling
        out = self.fc(x)
        return out


class CBAMShuffleNetV2(nn.Module):
    """
    Proposed Model: Knowledge-Distilled CBAM-Enhanced ShuffleNetV2.
    Integrates CBAM Attention modules after ShuffleNet Stages 2, 3, and 4
    to focus feature extraction on disease-affected leaf regions.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(CBAMShuffleNetV2, self).__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        base_model = shufflenet_v2_x1_0(weights=weights)

        self.conv1 = base_model.conv1
        self.maxpool = base_model.maxpool

        # Stage 2 (116 channels)
        self.stage2 = base_model.stage2
        self.cbam2 = CBAM(in_channels=116, reduction_ratio=16)

        # Stage 3 (232 channels)
        self.stage3 = base_model.stage3
        self.cbam3 = CBAM(in_channels=232, reduction_ratio=16)

        # Stage 4 (464 channels)
        self.stage4 = base_model.stage4
        self.cbam4 = CBAM(in_channels=464, reduction_ratio=16)

        self.conv5 = base_model.conv5

        # Final Classifier Head
        in_features = base_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

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
        x = x.mean([2, 3])  # Global Average Pooling
        out = self.fc(x)
        return out
