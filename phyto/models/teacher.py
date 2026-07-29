import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class TeacherResNet50(nn.Module):
    """
    High-Performance Teacher Model based on ResNet50.
    Provides dark knowledge logit distributions for student distillation.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(TeacherResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
