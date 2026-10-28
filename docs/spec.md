# 仕様

設定の出典は Nanda et al.（2023）の設定の章と、実装ガイド（grokking 実装ガイド）です。数値はすべて `configs/base.json` と一致させています。

## 入力と出力

課題は Z_113 上の加法 (a, b) ↦ (a + b) mod 113 です。入力は長さ3の系列 `[a, b, =]` で、`=` のトークン ID は `p`（113）、語彙は `p + 1` 個です。正解は `(a + b) mod p` の1クラス（p 通り）です。

| テンソル | 形 | dtype |
|---|---|---|
| `tokens` | `(p*p, 3)` = `(12769, 3)` | int64 |
| `labels` | `(p*p,)` | int64 |

行 n は a = n // p、b = n % p に対応します。全組の30%（3,830組）を訓練、残り（8,939組）をテストにします。分割は `data_seed` で初期化した専用の `torch.Generator` で決め、グローバルな乱数には触れません。

## モデル

LayerNorm とバイアスのない1層 Transformer です。d = 128、H = 4、d_head = 32、d_mlp = 512。

```
x⁽⁰⁾ᵢ = W_E[:, tᵢ] + W_pos[:, i]
Aʰᵢⱼ = softmax_{j≤i}( <W_Qʰ x⁽⁰⁾ᵢ, W_Kʰ x⁽⁰⁾ⱼ> / √d_head ),   x⁽¹⁾ᵢ = x⁽⁰⁾ᵢ + Σₕ W_Oʰ Σⱼ Aʰᵢⱼ W_Vʰ x⁽⁰⁾ⱼ
n = ReLU(W_in x⁽¹⁾₂),   x⁽²⁾ = x⁽¹⁾₂ + W_out n
ℓ = W_U x⁽²⁾ ∈ ℝᵖ
```

| 重み | 形 | 重み | 形 |
|---|---|---|---|
| `W_E` | `(d, p+1)` | `W_O` | `(H, d, d_head)` |
| `W_pos` | `(d, 3)` | `W_in` | `(d_mlp, d)` |
| `W_Q`, `W_K`, `W_V` | `(H, d_head, d)` | `W_out` | `(d, d_mlp)` |
| | | `W_U` | `(p, d)` |

初期値は N(0, 1/fan_in) で、`init_seed` で初期化した専用の `Generator` から作ります。パラメータ数は 226,048 です。因果マスクは buffer、中間値の取り出し口は `nn.Identity` の hook 点（6か所）です。`Transformer` は既定値を持たず、値はすべて `Config` から渡します。`seq_len` は `Config` に置かず、データの形（`tokens.shape[1]`）から決めます。

## 損失

`=` の位置（i = 2）の出力に対する交差エントロピーです。答えを出すのは最後の位置だけで、1層モデルではその位置の出力は他の位置の MLP に依存しないため、MLP も位置 2 だけで計算します。

## 最適化

AdamW（lr = 1e-3、weight_decay = 1.0、betas = (0.9, 0.98)）で、フルバッチ 40,000 ステップです。`torch.optim.AdamW` の weight_decay の既定値は 0.01 なので、必ず明示します。`torch.optim.Adam` の weight_decay は L2 正則化で挙動が違うため使いません。損失は float64 で計算します（`grokking/train.py` の `cross_entropy`）。float32 では、自信の強い予測に対する損失が約 1.2e-7 の倍数に丸められるためです。

## 公開実装との違い

Nanda et al. の公開実装（[progress-measures-paper](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper) の `transformers.py` と `helpers.py`、`main` ブランチ）と比べました。`p`、`train_frac`、d、H、d_mlp、lr、weight_decay、betas、フルバッチ、`W_E`・`W_pos`・`W_Q`・`W_K`・`W_V`・`W_O`・`W_in` の初期化、因果マスク、`=` の位置だけで損失を取る点は同じです。違いは次のとおりです。論文本文とはまだ照合していません。

| 項目 | 公開実装 | このリポジトリ |
|---|---|---|
| 学習ステップ数 | 50,000（`num_epochs`） | 40,000 |
| 学習率のウォームアップ | 最初の 10 ステップで線形に上げる（`LambdaLR`、`min(step/10, 1)`） | なし |
| MLP のバイアス | `b_in`、`b_out` あり（0 で初期化） | なし |
| `W_out` の初期化 | N(0, 1/d) | N(0, 1/d_mlp) |
| 出力の次元と `W_U` の初期化 | p + 1（`=` を含む）、N(0, 1/(p+1)) | p、N(0, 1/d) |
| 損失の精度 | コメントは float64 とするが、コードは float32 に変換している | float64 |
| 訓練データの選び方 | Python の `random.shuffle`（`seed`） | `torch.randperm`（`data_seed`） |

これらの違いが学習曲線にどれだけ影響するかは、まだ確かめていません。

## 記録するもの

`log_every`（100）ステップごとに、訓練とテストの損失・正解率と、全パラメータの二乗和を `metrics.csv` に追記します。チェックポイント（`state_dict`）は `Config.checkpoint_steps()` の対数間隔のステップ（約40個、0 と 40,000 を含む）で `ckpt_XXXXXX.pt` に保存します。`config.json` には設定と git のコミットハッシュを書きます。同じ `run_name` の出力先が空でなければ、学習を始めずに止めます。

## 完了条件

図1で、訓練精度が先に 100% になり、遅れてテスト精度が跳ねること。図2で、学習後の埋め込みスペクトルに少数の周波数だけが立つこと。図3で、MLP のニューロン活性が (a, b) に対して周期的な縞を示すこと。