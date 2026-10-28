import math
import torch
from grokking.analysis.fourier import fourier_basis, neuron_logit_spectrum, spectrum, top_frequencies

p = 7
B = fourier_basis(p)


def test_orthonormal():
    assert torch.allclose(B @ B.T, torch.eye(p), atol=1e-5)


def test_parseval():
    v = torch.randn(p, 3)
    assert torch.allclose((B @ v).norm(), v.norm(), atol=1e-5)


def test_pure_cosine_has_single_peak():
    x = torch.arange(p, dtype=torch.float32)
    f = torch.cos(2 * math.pi * 2 * x / p + 0.7).view(p, 1)      # 周波数 2、位相つき
    spec = spectrum(f, B)
    assert top_frequencies(spec, 1) == [2]
    assert spec[1] > 10 * spec[[0, 2]].max()

def test_neuron_logit_spectrum_ignores_eq_row():
    """W_U が (p+1, d) でも、数字の行だけを使って (K,) を返す。'=' の行は結果に効かない。"""
    W_U, W_out = torch.randn(p + 1, 4), torch.randn(4, 5)
    spec = neuron_logit_spectrum(W_U, W_out, p)
    assert spec.shape == (p // 2,)
    W_U[p] = 100.0
    assert torch.allclose(neuron_logit_spectrum(W_U, W_out, p), spec)
