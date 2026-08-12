"""
Knowledge Distillation Training Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Implements DistillationLoss (Combining Cross Entropy and KL Divergence)
and train_distillation_model() for transferring teacher representations to lightweight student models.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.feature_kd import FeatureKDAdapter, FeatureDistillationLoss
from src.training.train import load_checkpoint


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    1. Hard Cross-Entropy (ground-truth labels)
    2. Soft KL Divergence (teacher logits scaled by temperature T)
    3. Intermediate Feature Loss (projection adapted feature map matching)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        feature_weight: float = 0.0,
        feature_loss_type: str = "mse",
        feature_adapter: Optional[FeatureKDAdapter] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.feature_adapter = feature_adapter
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.feature_loss_fn = FeatureDistillationLoss(loss_type=feature_loss_type)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates combined distillation loss.
        """
        hard_loss = self.ce_loss(student_logits, labels)

        if self.alpha > 0.0:
            soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
            distill_loss = self.kl_div_loss(soft_student, soft_teacher) * (self.temperature ** 2)
            total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * distill_loss
        else:
            total_loss = hard_loss

        if self.feature_weight > 0.0 and student_feat is not None and teacher_feat is not None:
            if self.feature_adapter is not None:
                teacher_feat, student_feat = self.feature_adapter(teacher_feat, student_feat)
            feat_loss = self.feature_loss_fn(teacher_feat, student_feat)
            total_loss = total_loss + self.feature_weight * feat_loss

        return total_loss



def train_distillation_model(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Optional[Union[torch.device, str]] = None,
    epochs: int = 15,
    alpha: float = 0.7,
    temperature: float = 4.0,
    feature_weight: float = 0.0,
    feature_loss_type: str = "mse",
    feature_adapter: Optional[FeatureKDAdapter] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    last_checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict[str, List[float]]:
    """
    Executes Knowledge Distillation training (Logit KD, Feature KD, or Combined KD).
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    student_model = student_model.to(target_device)
    teacher_model = teacher_model.to(target_device)
    teacher_model.eval()

    if feature_adapter is not None:
        feature_adapter = feature_adapter.to(target_device)

    kd_criterion = DistillationLoss(
        alpha=alpha,
        temperature=temperature,
        feature_weight=feature_weight,
        feature_loss_type=feature_loss_type,
        feature_adapter=feature_adapter,
    )
    val_criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    best_val_acc: float = -1.0
    start_epoch: int = 1

    if resume_from_checkpoint is not None:
        (
            student_model,
            optimizer_loaded,
            scheduler_loaded,
            completed_epoch,
            loaded_best_val_acc,
            loaded_history,
        ) = load_checkpoint(
            checkpoint_path=resume_from_checkpoint,
            model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=target_device,
        )
        if optimizer_loaded is not None:
            optimizer = optimizer_loaded
        if scheduler_loaded is not None:
            scheduler = scheduler_loaded

        start_epoch = completed_epoch + 1
        best_val_acc = loaded_best_val_acc
        history = loaded_history
        print(f"Resuming distillation training from epoch {start_epoch} (completed {completed_epoch}/{epochs}).")

    if start_epoch > epochs:
        print(f"Distillation training already completed ({start_epoch - 1}/{epochs} epochs). Returning history.")
        return history

    target_last_path: Optional[Path] = None
    if last_checkpoint_path is not None:
        target_last_path = Path(last_checkpoint_path)
    elif checkpoint_path is not None:
        cp = Path(checkpoint_path)
        target_last_path = cp.with_name(f"{cp.stem}_last{cp.suffix}")

    use_features = feature_weight > 0.0

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.perf_counter()

        # Training Phase
        student_model.train()
        if feature_adapter is not None:
            feature_adapter.train()

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            with torch.no_grad():
                if use_features:
                    teacher_logits, teacher_feat = teacher_model(images, return_features=True)
                else:
                    teacher_logits = teacher_model(images)
                    teacher_feat = None

            optimizer.zero_grad()
            if use_features:
                student_logits, student_feat = student_model(images, return_features=True)
            else:
                student_logits = student_model(images)
                student_feat = None

            loss = kd_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_feat=student_feat,
                teacher_feat=teacher_feat,
            )
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_train_loss += float(loss.item()) * batch_size
            _, preds = torch.max(student_logits, 1)
            correct_train += int((preds == labels).sum().item())
            total_train += batch_size

        epoch_train_loss = running_train_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = (correct_train / total_train) * 100.0 if total_train > 0 else 0.0

        # Validation Phase
        student_model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(target_device)
                labels = labels.to(target_device)

                outputs = student_model(images)
                loss = val_criterion(outputs, labels)

                batch_size = images.size(0)
                running_val_loss += float(loss.item()) * batch_size
                _, preds = torch.max(outputs, 1)
                correct_val += int((preds == labels).sum().item())
                total_val += batch_size

        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0.0
        epoch_val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0

        current_lr = float(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        elapsed_time = time.perf_counter() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["validation_loss"].append(epoch_val_loss)
        history["validation_accuracy"].append(epoch_val_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(elapsed_time)

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc

        checkpoint_data: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_validation_accuracy": best_val_acc,
            "training_history": history,
            "distillation_config": {
                "alpha": alpha,
                "temperature": temperature,
                "feature_weight": feature_weight,
                "feature_loss_type": feature_loss_type,
            },
        }

        if is_best and checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            ckpt_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, ckpt_p)

        if target_last_path is not None:
            target_last_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, target_last_path)

        best_flag = " [BEST]" if is_best else ""
        print(
            f"Distill Epoch [{epoch:02d}/{epochs:02d}] - "
            f"Time: {elapsed_time:.2f}s - LR: {current_lr:.6f} - "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%{best_flag}"
        )

    return history
