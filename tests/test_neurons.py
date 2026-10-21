import torch

from grokking.analysis.neurons import capture, mlp_activations
from grokking.model import Transformer


p = 7
model = Transformer(p, d_model=16, n_heads=2, d_mlp=8, seq_len=3, init_seed=0)


def test_mlp_activations_shape():
    """全組の活性が (p, p, d_mlp) で返る。"""
    assert mlp_activations(model).shape == (p, p, 8)


def test_mlp_activations_axes_are_a_then_b():
    """acts[a, b] が、入力 [a, b, '='] を1組だけ流したときの活性に一致する（軸の順が a, b）。"""
    acts = mlp_activations(model)
    single = capture(model, model.hook_mlp_act, torch.tensor([[3, 5, p]]))[0]
    assert torch.allclose(acts[3, 5], single, atol=1e-5)
    assert not torch.allclose(acts[3, 5], acts[5, 3])      # a と b を取り違えると検出できる