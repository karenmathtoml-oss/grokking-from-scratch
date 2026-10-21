from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """入力 [a, b, =] を受け取り、'=' の位置で (a + b) mod p のロジットを出力する1層モデル。"""

    def __init__(self, p: int, d_model: int, n_heads: int, d_mlp: int, seq_len: int, init_seed: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model は n_heads で割り切れる必要がある"
        self.p = p
        self.d_head = d_model // n_heads
        g = torch.Generator().manual_seed(init_seed)

        def param(*shape: int, fan_in: int) -> nn.Parameter:
            """N(0, 1/fan_in) で初期化した学習パラメータ。"""
            w = torch.randn(*shape, generator=g)
            w = w / math.sqrt(fan_in)
            return nn.Parameter(w)

        self.W_E = param(d_model, p + 1, fan_in=d_model)                  # (d, p+1)
        self.W_pos = param(d_model, seq_len, fan_in=d_model)              # (d, S)
        self.W_Q = param(n_heads, self.d_head, d_model, fan_in=d_model)   # (H, d_head, d)
        self.W_K = param(n_heads, self.d_head, d_model, fan_in=d_model)   # (H, d_head, d)
        self.W_V = param(n_heads, self.d_head, d_model, fan_in=d_model)   # (H, d_head, d)
        self.W_O = param(n_heads, d_model, self.d_head, fan_in=d_model)   # (H, d, d_head)
        self.W_in = param(d_mlp, d_model, fan_in=d_model)                 # (d_mlp, d)
        self.W_out = param(d_model, d_mlp, fan_in=d_mlp)                  # (d, d_mlp)
        self.W_U = param(p, d_model, fan_in=d_model)                      # (p, d)

        # Causal mask
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask)
        mask = mask.view(1, 1, seq_len, seq_len)                          # (1, 1, S, S)
        self.register_buffer("mask", mask)
 
        # hook 点（forward hook を掛けると中間値を取り出せる）
        self.hook_embed = nn.Identity()          # x^(0)            (B, S, d)
        self.hook_attn_pattern = nn.Identity()   # A^h_ij           (B, H, S, S)
        self.hook_attn_out = nn.Identity()       # Attention の出力 (B, S, d)
        self.hook_resid_mid = nn.Identity()      # x^(1)_2          (B, d)
        self.hook_mlp_act = nn.Identity()        # n = ReLU(.)      (B, d_mlp)
        self.hook_resid_post = nn.Identity()     # x^(2)            (B, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: LongTensor (B, S) → logits (B, p)。'=' だけを出力する。"""
        S = tokens.shape[1]

        # Embedding
        x0 = self.W_E[:, tokens]                                 # (d, B, S)
        x0 = x0.permute(1, 2, 0)                                 # (B, S, d)
        x0 = x0 + self.W_pos[:, :S].T                            # 位置埋め込みを加える
        x0 = self.hook_embed(x0)

        # Attention（b:バッチ, s/q/k:位置, h:ヘッド, m:残差次元, d:ヘッド次元）
        q = torch.einsum("bsm,hdm->bhsd", x0, self.W_Q)          # (B, H, S, d_head)
        k = torch.einsum("bsm,hdm->bhsd", x0, self.W_K)          # (B, H, S, d_head)
        v = torch.einsum("bsm,hdm->bhsd", x0, self.W_V)          # (B, H, S, d_head)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)           # (B, H, S, S)
        scores = scores / math.sqrt(self.d_head)
        future = self.mask[:, :, :S, :S] == 0                    # 未来の位置
        scores = scores.masked_fill(future, float("-inf"))
        pattern = F.softmax(scores, dim=-1)
        pattern = self.hook_attn_pattern(pattern)
        z = torch.einsum("bhqk,bhkd->bhqd", pattern, v)          # (B, H, S, d_head)
        attn_out = torch.einsum("bhqd,hmd->bqm", z, self.W_O)    # (B, S, d)
        attn_out = self.hook_attn_out(attn_out)
        x1 = x0 + attn_out                                       # 残差接続

        # MLP
        x1_last = x1[:, -1, :]                                   # (B, d)
        x1_last = self.hook_resid_mid(x1_last)
        n = x1_last @ self.W_in.T                                # (B, d_mlp)
        n = F.relu(n)
        n = self.hook_mlp_act(n)
        mlp_out = n @ self.W_out.T                               # (B, d)
        x2 = x1_last + mlp_out                                   # 残差接続
        x2 = self.hook_resid_post(x2)

        return x2 @ self.W_U.T                                   # (B, p)

    def weight_sq_norm(self) -> float:
        """全パラメータの二乗和 Σ θ_i^2"""
        sq_sums = [(prm ** 2).sum().item() for prm in self.parameters()]
        return sum(sq_sums)