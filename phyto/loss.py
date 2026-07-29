import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss function.
    Combines:
    1. Cross-Entropy Hard Loss (Student predictions vs True ground-truth labels)
    2. KL-Divergence Soft Loss (Student softened logits vs Teacher softened logits at temperature T)
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard Cross-Entropy Hard Loss
        loss_ce = self.ce_loss(student_logits, targets)

        # Softened probabilities with Temperature T
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL Divergence Soft Loss scaled by T^2
        loss_kd = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Total Composite Loss
        total_loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd
        return total_loss
