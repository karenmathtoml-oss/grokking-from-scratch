import torch
import torch.nn.functional as F

from grokking.train import cross_entropy


def test_cross_entropy_keeps_tiny_losses():
    """自信の強い予測でも損失が 0 に潰れない（float32 では 0 になる）。"""
    logits = torch.zeros(1, 7)
    logits[0, 3] = 40.0
    labels = torch.tensor([3])
    expected = 6 * torch.exp(torch.tensor(-40.0, dtype=torch.float64))   # ≈ log(1 + 6e^-40)
    assert F.cross_entropy(logits, labels).item() == 0.0
    assert torch.isclose(cross_entropy(logits, labels), expected, rtol=1e-6)


def test_cross_entropy_matches_float32_on_ordinary_logits():
    logits = torch.randn(5, 7)
    labels = torch.randint(0, 7, (5,))
    assert torch.isclose(cross_entropy(logits, labels).float(), F.cross_entropy(logits, labels), atol=1e-6)
