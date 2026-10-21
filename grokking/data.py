from __future__ import annotations

import torch


class CyclicGroup:
    """位数 p の巡回群を表すクラス。"""

    def __init__(self, p: int):
        """位数 p を指定して巡回群を初期化する。"""
        self.p = p

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """群演算 a + b mod p を要素ごとに計算する。"""
        return (a + b) % self.p

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        """ a の逆元 (-a) mod p を計算する。"""
        return (-a) % self.p


def make_dataset(group: CyclicGroup) -> tuple[torch.Tensor, torch.Tensor]:
    """群演算 a + b mod p を要素ごとに計算し、 p*p 組の演算表データを作成する。"""
    p = group.p
    a = torch.arange(p).unsqueeze(1).repeat(1, p).view(-1)
    b = torch.arange(p).repeat(p)
    tokens = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    labels = group.op(a, b)

    return tokens, labels

def train_test_split(tokens, labels, train_frac: float, seed: int):
    """演算表データをランダムにシャッフルし、seedにより固定。訓練データとテストデータに分割する。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(tokens), generator=g)

    train_size = int(len(tokens) * train_frac)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    return tokens[train_idx], labels[train_idx], tokens[test_idx], labels[test_idx]