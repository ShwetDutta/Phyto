import unittest
import torch
from phyto.models.cbam import CBAM, ChannelAttention, SpatialAttention
from phyto.models.shufflenet_v2 import BaselineShuffleNetV2, CBAMShuffleNetV2
from phyto.models.teacher import TeacherResNet50
from phyto.loss import KnowledgeDistillationLoss

class TestPhytoModels(unittest.TestCase):

    def test_cbam_forward(self):
        x = torch.randn(2, 64, 28, 28)
        cbam = CBAM(in_channels=64, reduction_ratio=16)
        out = cbam(x)
        self.assertEqual(out.shape, x.shape, "CBAM output shape mismatch")

    def test_baseline_shufflenet_forward(self):
        x = torch.randn(2, 3, 224, 224)
        model = BaselineShuffleNetV2(num_classes=5, pretrained=False)
        out = model(x)
        self.assertEqual(out.shape, (2, 5), "Baseline ShuffleNetV2 output shape mismatch")

    def test_cbam_shufflenet_forward(self):
        x = torch.randn(2, 3, 224, 224)
        model = CBAMShuffleNetV2(num_classes=5, pretrained=False)
        out = model(x)
        self.assertEqual(out.shape, (2, 5), "CBAM ShuffleNetV2 output shape mismatch")

    def test_teacher_resnet_forward(self):
        x = torch.randn(2, 3, 224, 224)
        teacher = TeacherResNet50(num_classes=5, pretrained=False)
        out = teacher(x)
        self.assertEqual(out.shape, (2, 5), "Teacher ResNet50 output shape mismatch")

    def test_kd_loss_calculation(self):
        student_logits = torch.randn(4, 5, requires_grad=True)
        teacher_logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])

        kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
        loss = kd_loss(student_logits, teacher_logits, targets)
        
        self.assertTrue(torch.is_tensor(loss), "KD Loss is not a tensor")
        self.assertGreater(loss.item(), 0.0, "KD Loss must be positive")
        loss.backward()
        self.assertIsNotNone(student_logits.grad, "Gradients not computed for student logits")

if __name__ == "__main__":
    unittest.main()
