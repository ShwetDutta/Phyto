"""
Feature Knowledge Distillation Adapter and Loss Modules for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Implements spatial and channel feature map adaptation between teacher models
(e.g., EfficientNet-B0: 1280 channels / 8x8) and student models (e.g., ShuffleNetV2 x0.5: 1024 channels / 8x8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDAdapter(nn.Module):
    """
    1x1 Conv + BatchNorm projection adapter to align teacher and student intermediate feature representations.
    Maps teacher feature channels to student feature channels.
    """

    def __init__(self, teacher_channels: int = 1280, student_channels: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, student_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(student_channels),
        )

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns teacher features spatially and dimensionally with student features.
        """
        # Align spatial dimensions if they differ
        if teacher_feat.shape[2:] != student_feat.shape[2:]:
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])

        # Project teacher channels down to student channel space
        projected_teacher_feat = self.proj(teacher_feat)
        return projected_teacher_feat, student_feat


class FeatureDistillationLoss(nn.Module):
    """
    Computes feature-level distillation loss between normalized intermediate feature representations.
    Supports Normalized MSE Loss and Cosine Distance Loss.
    """

    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.mse = nn.MSELoss()

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        Computes feature loss between aligned teacher and student feature maps.
        """
        # Normalize features across channel dimension (L2 normalization)
        norm_t = F.normalize(teacher_feat, p=2, dim=1)
        norm_s = F.normalize(student_feat, p=2, dim=1)

        if self.loss_type == "cosine":
            # Cosine similarity loss: 1 - mean cosine similarity
            cosine_sim = F.cosine_similarity(norm_t, norm_s, dim=1)
            return (1.0 - cosine_sim.mean())
        else:
            # Normalized MSE Loss
            return self.mse(norm_s, norm_t)


__all__ = ["FeatureKDAdapter", "FeatureDistillationLoss"]
