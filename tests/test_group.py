import torch

from grokking.data import CyclicGroup, make_dataset, train_test_split


p = 7
G = CyclicGroup(p)
x = torch.arange(p)
a, b, c = x.unsqueeze(1).repeat(1, p).view(-1), x.repeat(p), torch.full_like(x.repeat(p), p)

def test_closure():
    """群の閉包性を確認する。"""
    r = G.op(a, b)
    assert torch.all((r >= 0) & (r < p))

def test_associativity():
    """群の結合律を確認する。"""
    ## r1 = G.op(G.op(a, b), c)
    ## r2 = G.op(a, G.op(b, c))
    ## assert torch.all(r1 == r2)

    a3, b3, c3 = torch.meshgrid(x, x, x, indexing="ij")
    a3, b3, c3 = a3.reshape(-1), b3.reshape(-1), c3.reshape(-1)   # 各 (p**3,)
    assert torch.equal(G.op(G.op(a3, b3), c3), G.op(a3, G.op(b3, c3)))
    ## 結合律が全パターンで成り立つことを確認するため、便宜的に torch.meshgrid を使って a, b, c の全組み合わせを生成している。

def test_identity():
    """群の単位元を確認する。"""
    r = G.op(a, torch.zeros_like(a))
    assert torch.all(r == a)

def test_inverse():
    """群の逆元を確認する。"""
    r = G.op(a, G.inverse(a))
    assert torch.all(r == 0)

def test_commutativity():
    """巡回群の可換性を確認する。"""
    r1 = G.op(a, b)
    r2 = G.op(b, a)
    assert torch.all(r1 == r2)