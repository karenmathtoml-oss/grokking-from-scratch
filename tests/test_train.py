from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.config import Config
from grokking.train import cross_entropy, train, warmup_factor

ROOT = Path(__file__).resolve().parent.parent


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


def test_warmup_factor():
    """公開実装の min(step/10, 1) と同じ係数。warmup_steps=0 なら常に 1。"""
    assert [warmup_factor(s, 10) for s in range(12)] == [s / 10 for s in range(11)] + [1.0]
    assert warmup_factor(0, 0) == 1.0


def test_first_update_has_zero_learning_rate(tmp_path):
    """ウォームアップの最初の更新は学習率 0 なので、重みが変わらない。2回目からは変わる。"""
    cfg = Config.load(ROOT / "configs" / "base.json", p=7, d_model=16, n_heads=2, d_mlp=8,
                      steps=2, log_every=1, n_checkpoints=3, run_name="warmup")
    run_dir = train(cfg, root=tmp_path, device="cpu")
    w0, w1, w2 = (torch.load(run_dir / f"ckpt_{s:06d}.pt") for s in (0, 1, 2))
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert not torch.equal(w1["W_E"], w2["W_E"])
