# grokking-from-scratch

[English](README.md) | 日本語

Nanda et al.（2023）*Progress measures for grokking via mechanistic interpretability* を、一から再現するリポジトリです。1層の Transformer に、113 を法とする加法を、全組の30%だけで学習させます。モデルは数百ステップで訓練データを丸暗記し、テストデータへの汎化は、それよりずっと後に急に起こります（grokking）。学習したネットワークが、少数の周波数を使って `a + b` を計算していることを示すフーリエ解析も含みます。

**現状：** データ、モデル、設定、学習ループ、フーリエ解析、図の作成のコードはすべて実装済みで、テストで確認しています。40,000 ステップの本番の学習と、最終的な図はこれからです。

## セットアップ

コマンドはすべて、リポジトリの直下で実行します。

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install torch numpy pandas matplotlib imageio pytest
python -m pytest tests -q
```

## 学習

```bash
python -m grokking.train --config configs/base.json
```

本番の学習は、Google Colab の GPU で行う想定です（`notebooks/01_train.ipynb`）。CPU では、2,000 ステップのスモーク実行（`configs/experiments/smoke.json`）に 242 秒、1ステップあたり約 0.12 秒かかりました。

実行の結果は `runs/<run_name>/` に書かれます。このフォルダは git の管理外です。

| ファイル | 内容 |
|---|---|
| `config.json` | 全設定と、コードの git コミットハッシュ |
| `metrics.csv` | `step, train_loss, train_accuracy, test_loss, test_accuracy, weight_sq_norm`（`log_every` ステップごと） |
| `ckpt_XXXXXX.pt` | 対数間隔のステップでのモデルの `state_dict`（1回の実行で約40個） |

## 設定

`configs/base.json` は論文の設定です。既定値は `grokking/config.py` の `Config` だけに書き、両者が一致することをテストで確かめています。

| キー | 値 | キー | 値 |
|---|---|---|---|
| `p` | 113 | `lr` | 1e-3 |
| `train_frac` | 0.3 | `weight_decay` | 1.0 |
| `d_model` | 128 | `betas` | (0.9, 0.98) |
| `n_heads` | 4 | `steps` | 40,000（フルバッチ、AdamW） |
| `d_mlp` | 512 | `log_every` / `n_checkpoints` | 100 / 40 |

`data_seed` が訓練/テストの分割を、`init_seed` が重みの初期値を決め、2つは互いに独立です。実験は、基準設定との差分だけを書いた小さな JSON で表します。知らないキーがあると、エラーになります。

```json
{"base": "../base.json", "weight_decay": 0.0, "run_name": "A_wd_0"}
```

## 解析と図

`notebooks/02_fourier_analysis.ipynb` は、チェックポイントを読み込み、埋め込み `W_E` と、ニューロンからロジットへの写像 `W_U W_out` のフーリエスペクトルを計算し、`grokking/analysis/plots.py` で図を作ります。

| 図 | 内容 |
|---|---|
| `figures/fig1_curve.png` | 損失と精度の曲線（横軸は対数） |
| `figures/fig2_spectrum.gif` | チェックポイントごとの埋め込みスペクトルの変化 |
| `figures/fig3_neuron.png` | 1つの MLP ニューロンの、全 `(a, b)` に対する活性 |

図は、本番の学習のあとに追加します。

<!-- ![図1](figures/fig1_curve.png) -->
<!-- ![図2](figures/fig2_spectrum.gif) -->
<!-- ![図3](figures/fig3_neuron.png) -->

## ディレクトリ構成

```
grokking/            data.py, model.py, config.py, train.py
grokking/analysis/   fourier.py, neurons.py, plots.py
configs/             base.json, experiments/*.json
notebooks/           01_train.ipynb, 02_fourier_analysis.ipynb（出力はクリア済み）
tests/               pytest のテスト
docs/                spec.md（仕様）、notes.md（詰まったところと判断の記録）
```

## 参考文献

Nanda, Chan, Lieberum, Smith, Steinhardt (2023). Progress measures for grokking via mechanistic interpretability. ICLR 2023. arXiv:2301.05217.

Power, Burda, Edwards, Babuschkin, Misra (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

Liu, Michaud, Tegmark (2023). Omnigrok: Grokking beyond algorithmic data. arXiv:2210.01117.

Loshchilov, Hutter (2019). Decoupled weight decay regularization. arXiv:1711.05101.

## ライセンス

`LICENSE` を参照してください。