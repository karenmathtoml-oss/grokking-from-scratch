from __future__ import annotations

import math
import torch


def fourier_basis(p: int) -> torch.Tensor:
    """正規直交基底 F(p, p)。行の並びは e_0, c_1, s_1, c_2, s_2, ..., c_K, s_K（K = (p-1)/2）。"""
    assert p % 2 == 1
    x = torch.arange(p, dtype=torch.float32)
    rows = [torch.ones(p) / math.sqrt(p)]
    for k in range(1, p // 2 + 1):
        angle = 2 * math.pi * k * x / p
        rows.append(torch.cos(angle) / math.sqrt(p / 2))
        rows.append(torch.sin(angle) / math.sqrt(p / 2))
    return torch.stack(rows)

@torch.no_grad() # 勾配の計算を記録しない
def spectrum(W: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """W の各列（Z_p 上の関数とみなす。W は (p, n)）を基底で展開し、周波数ごとの強さを返す。
    返り値 (K,)：周波数 k の強さ = sqrt(‖cos 係数‖² + ‖sin 係数‖²)。位相に依らない量。
    """
    coef = basis @ W                                   # (p, n)  行 j が基底 j の係数
    cos_norm = coef[1::2].norm(dim=1)                  # (K,)  c_k の係数ベクトルのノルム
    sin_norm = coef[2::2].norm(dim=1)
    return torch.sqrt(cos_norm ** 2 + sin_norm ** 2)

@torch.no_grad()
def embedding_spectrum(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """埋め込み W_E (d, p+1) のうち数字 0..p-1 の列だけを使う。"""
    return spectrum(W_E[:, :p].T, fourier_basis(p).to(W_E.device))

@torch.no_grad()
def neuron_logit_spectrum(W_U: torch.Tensor, W_out: torch.Tensor, p: int) -> torch.Tensor:
    """W_L = W_U W_out (p, d_mlp) の列を展開する。"""
    return spectrum(W_U @ W_out, fourier_basis(p).to(W_U.device))


def top_frequencies(spec: torch.Tensor, n: int = 5) -> list[int]:
    return (spec.argsort(descending=True)[:n] + 1).tolist()   # 添字 0 が k = 1

@torch.no_grad()
def circle_projection(W_E: torch.Tensor, p: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """周波数 k の cos / sin 方向に各トークンの埋め込みを射影する。返り値は各 (p,)。"""
    basis = fourier_basis(p).to(W_E.device)
    W = W_E[:, :p]                                     # (d, p)
    u = W @ basis[2 * k - 1]
    u = u / u.norm()
    v = W @ basis[2 * k]
    v = v / v.norm()
    return u @ W, v @ W