import math
import torch
from grokking.analysis.fourier import fourier_basis, spectrum, top_frequencies

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