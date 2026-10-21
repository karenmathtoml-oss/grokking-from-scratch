import pytest
import torch
import torch.nn.functional as F

from grokking.analysis.neurons import capture
from grokking.model import Transformer

p = 7
tokens = torch.tensor([[1, 2, p], [3, 4, p]])
 
 
def make_model(init_seed: int = 0) -> Transformer:
    """テスト用の小さいモデル。Config の値には依存しない。"""
    return Transformer(p, d_model=16, n_heads=2, d_mlp=8, seq_len=3, init_seed=init_seed)
 
 
model = make_model()


def test_output_shape_and_dtype():
    """出力は (B, p) の float32 ロジットで、すべて有限値。"""
    logits = model(tokens)
    assert logits.shape == (2, p)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

def test_init_seed():
    """全パラメータが init_seed だけで決まり、グローバル乱数に影響されない。"""
    torch.manual_seed(1)
    m1 = make_model(init_seed=0)
    torch.manual_seed(2)
    m2 = make_model(init_seed=0)
    m3 = make_model(init_seed=1)
    for (name, a), b, c in zip(m1.named_parameters(), m2.parameters(), m3.parameters()):
        assert torch.equal(a, b), name
        assert not torch.equal(a, c), name

def test_parameter_count():
    """既定設定（p = 113）のパラメータ数が 226,048（各重みの形が仕様どおり）。"""
    n = sum(prm.numel() for prm in make_model(init_seed=0).parameters())
    assert n == 226_048

def test_causal_mask():
    """未来の位置への注意が 0 で、各行の和が 1。"""
    pattern = capture(model, model.hook_attn_pattern, tokens)   # (B, H, S, S)
    assert (pattern[..., 0, 1:] == 0).all()
    assert (pattern[..., 1, 2] == 0).all()
    row_sums = pattern.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))

@pytest.mark.parametrize("hook_name, shape", [
    ("hook_embed", (2, 3, 16)),
    ("hook_attn_pattern", (2, 2, 3, 3)),
    ("hook_attn_out", (2, 3, 16)),
    ("hook_resid_mid", (2, 16)),
    ("hook_mlp_act", (2, 8)),
    ("hook_resid_post", (2, 16)),
])

def test_hook_shapes(hook_name, shape):
    """6つの hook 点がすべて forward で通り、想定した形の値を返す。"""
    out = capture(model, getattr(model, hook_name), tokens)
    assert out.shape == shape

def test_batch_independence():
    """各問題の出力は、同じバッチに入っている他の問題に影響されない。"""
    full = model(tokens)
    single = model(tokens[:1])
    assert torch.allclose(full[:1], single, atol=1e-5)

def test_all_parameters_receive_gradient():
    """損失から全パラメータに勾配が届く（forward で使い忘れた重みがない）。"""
    m = make_model(init_seed=0)
    labels = (tokens[:, 0] + tokens[:, 1]) % p
    loss = F.cross_entropy(m(tokens), labels)
    loss.backward()
    for name, prm in m.named_parameters():
        assert prm.grad is not None, name
        assert prm.grad.abs().sum() > 0, name

def test_weight_sq_norm():
    """全パラメータを 1 にすると、二乗和はパラメータ数と一致する。"""
    m = make_model(init_seed=0)
    with torch.no_grad():
        for prm in m.parameters():
            prm.fill_(1.0)
    n = sum(prm.numel() for prm in m.parameters())
    assert m.weight_sq_norm() == n