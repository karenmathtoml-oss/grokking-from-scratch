import torch
from grokking.data import CyclicGroup, make_dataset, train_test_split


p = 7
G = CyclicGroup(p)
tokens, labels = make_dataset(G)

def test_shapes_and_dtypes():
    """データセットの形状とデータ型を確認する。"""
    assert tokens.shape == (p * p, 3)
    assert labels.shape == (p * p,)
    assert tokens.dtype == torch.int64
    assert labels.dtype == torch.int64

def test_train_test_split():
    """訓練・テストの分割（個数・重なり・対応）を確認する。"""
    train_frac = 0.8
    seed = 42
    tr_x, tr_y, te_x, te_y = train_test_split(tokens, labels, train_frac, seed)

    # 個数：訓練は int(N * train_frac)、テストはその残り
    assert len(tr_x) == int(len(tokens) * train_frac)
    assert len(te_x) == len(tokens) - len(tr_x)

    # 重なりなし、かつ合わせると全問題が揃う
    tr = {tuple(r) for r in tr_x.tolist()}
    te = {tuple(r) for r in te_x.tolist()}
    assert not (tr & te)
    assert len(tr | te) == p * p

    # tokens と labels の対応が保たれている（正解は (a + b) mod p）
    assert torch.equal((tr_x[:, 0] + tr_x[:, 1]) % p, tr_y)
    assert torch.equal((te_x[:, 0] + te_x[:, 1]) % p, te_y)

def test_hand_examples():
    """手計算で確認できる例を使って、データセットの正しさを確認する。"""
    assert labels[5 * p + 4].item() == (5 + 4) % p  # a=5, b=4 のときの正解
    assert labels[2 * p + 3].item() == (2 + 3) % p  # a=2, b=3 のときの正解
    assert labels[6 * p + 6].item() == (6 + 6) % p  # a=6, b=6 のときの正解
    assert labels[0 * p + 0].item() == (0 + 0) % p  # a=0, b=0 のときの正解

def test_all_pairs_once_and_eq_token():
    """全ての組み合わせが一度ずつ出現していること、tokens の最後の列が p であることを確認する。"""
    unique_pairs = {tuple(r) for r in tokens[:, :2].tolist()}
    assert len(unique_pairs) == p * p
    for a in range(p):
        for b in range(p):
            assert (a, b) in unique_pairs

    # tokens の最後の列が p であること
    assert torch.all(tokens[:, 2] == p)

def test_split_independent_of_global_rng():
    """分割とシャッフルが、グローバルな乱数生成器の状態に依存しないことを確認する。"""
    torch.manual_seed(1); a = train_test_split(tokens, labels, 0.3, seed=0)[0]
    torch.manual_seed(2); b = train_test_split(tokens, labels, 0.3, seed=0)[0]
    torch.manual_seed(3); c = train_test_split(tokens, labels, 0.3, seed=1)[0]

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    state = torch.get_rng_state()
    train_test_split(tokens, labels, 0.3, seed=0)
    assert torch.equal(state, torch.get_rng_state())