# grokking-from-scratch

Nanda et al.（2023）*Progress measures for grokking via mechanistic interpretability* を、一から再現するリポジトリです。1層Transformer に、113 を法とする加法を、全データセットの30%だけで学習させます。モデルは数百ステップで訓練データを丸暗記しますが、テストデータへの汎化は、それよりずっと後に急に起こります（grokking）。学習したネットワークが、少数の周波数を使って `a + b` を計算していることを示すフーリエ解析も行います。

**実行結果：** 基準設定で 40,000 ステップの本番の学習を行い、訓練データの暗記に対する汎化の遅れと、訓練後に少数の周波数に集中するフーリエ構造の両方を再現しました（[結果](#結果)）。

## セットアップ

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

依存パッケージの下限は `pyproject.toml` にあります。動作を確認した版は Python 3.13、torch 2.14.1、numpy 2.5.4、pandas 3.1.0、matplotlib 3.11.2、imageio 2.38.1（CPU）です。

## 学習

```bash
python -m grokking.train --config configs/base.json
```

本番の学習は、Google Colab の GPU で行う想定です（`notebooks/01_train.ipynb`）。
CPU では、2,000 ステップのスモーク実行（`configs/experiments/smoke.json`）に 約250秒掛かりました。この場合、本番実行には数十分掛かることが予想されるので、CPU環境はおすすめしません。

実行の結果は `runs/<run_name>/` に書かれます。このフォルダは git の管理外です。

| ファイル | 内容 |
|---|---|
| `config.json` | 全設定と、コードの git コミットハッシュ |
| `metrics.csv` | `step, train_loss, train_accuracy, test_loss, test_accuracy, weight_sq_norm`（`log_every` ステップごと） |
| `ckpt_XXXXXX.pt` | 対数間隔のステップでのモデルの `state_dict`（1回の実行で約40個） |

## 設定

`configs/base.json` のハイパーパラメータと、モデルの構造・初期化・学習率のウォームアップは、元論文の公開実装に合わせています。残る違いは、ステップ数（40,000）、損失の精度（float64）、訓練データの選び方の3つです（[`docs/spec.md` の「公開実装との違い」](docs/spec.md#公開実装との違い)）。設定の値はこのファイルにだけ書きます。`grokking/config.py` の `Config` は項目の名前と型だけを定め、既定値を持ちません。キーが足りなくても、知らないキーがあっても、読み込み時にエラーになります。

| キー | 値 | キー | 値 |
|---|---|---|---|
| `p` | 113 | `lr` | 1e-3 |
| `train_frac` | 0.3 | `weight_decay` | 1.0 |
| `d_model` | 128 | `betas` | (0.9, 0.98) |
| `n_heads` | 4 | `steps` | 40,000（フルバッチ、AdamW） |
| `d_mlp` | 512 | `log_every` / `n_checkpoints` | 100 / 40 |
| `warmup_steps` | 10 | | |

訓練データの割合は `train_frac` で決まります。`data_seed` はどの組を訓練データに入れるか（並べ替え）を、`init_seed` は重みの初期値を決め、2つは互いに独立です。実験は、基準設定との差分だけを書いた `configs/experiments/*.json` で表し、`"base"` キーで `base.json` を参照します。知らないキーがあると、エラーになります。

```json
{"base": "../base.json", "weight_decay": 0.0, "run_name": "A_wd_0"}
```

## 解析と図示

`notebooks/02_fourier_analysis.ipynb` はチェックポイントを読み込み、埋め込み `W_E` と、ニューロンからロジットへの写像 `W_U W_out` のフーリエスペクトルを計算します。この値を、`grokking/analysis/plots.py` で図示することができます。

| 図 | 内容 |
|---|---|
| `figures/fig1_curve.png` | 損失と精度の対数グラフ |
| `figures/fig2_spectrum.gif` | チェックポイントごとの埋め込みスペクトルの変化 |
| `figures/fig2_spectrum_before_after.png` | ステップ 0 の `W_E` と、学習後の `W_E`・`W_L` のスペクトル |
| `figures/fig3_neuron.png` | 1つの MLP ニューロンの、全データ `(a, b)` に対する活性 |

Google Colabでの実行時、ノートブックは図を Google Drive の `runs/base/figures/` に保存します。

## 結果

基準設定（`configs/base.json`、`data_seed = init_seed = 0`）で、Colab の T4 GPU を使い、フルバッチで 40,000 ステップ学習した1回の結果です。

この章の図と数値は、公開実装に合わせる前の版のコードによるものです。その版は、MLP のバイアスがなく、出力が p 次元で、`W_out`・`W_U` の初期化が異なり、学習率のウォームアップがなく、損失を float32 で計算していました。

### 図1：grokking の曲線

![損失と精度の曲線](figures/fig1_curve.png)

訓練精度はステップ 200 までに 100% になります。一方、テスト精度はステップ 4,000 前後まで約 10% 以下にとどまります。その後テスト精度は急に上がり、ステップ 7,000 ごろにはほぼ 100% に達します。テスト損失も、同じ区間で急激に下がります。

この結果は、損失を float32 で計算していた版のコードによるものです。訓練損失は 1e-7 付近で止まったあと何度も跳ね上がり、テスト精度もステップ 8,000 付近や学習の最後で一時的に下がっています。この振動は float32 の丸めが原因で、現在のコードは損失を float64 で計算します（[損失の精度と振動](#損失の精度と振動)）。

### 図2：埋め込みのフーリエスペクトル

![チェックポイントごとの埋め込みスペクトル](figures/fig2_spectrum.gif)

![学習前後のスペクトル](figures/fig2_spectrum_before_after.png)

ステップ 0 では、56 個の周波数がどれもほぼ同じ強さです。学習後は、埋め込み `W_E` が5つ前後の周波数に集中し、それ以外はほぼ 0 になります。ニューロンからロジットへの写像 `W_L = W_U W_out` は、そのうちの4つと同じ周波数にピークを持ちます。つまりネットワークは、入口でも出口でも共通の少数の周波数を使って `a + b` を扱っており、Nanda et al. の報告と一致します。

### 図3：MLP のニューロン

![ニューロンにおける規則性の観測](figures/fig3_neuron.png)

全入力に対する分散が最大のニューロンは、`(a, b)` 平面の上で規則的な格子状に発火します。各軸に沿って6周期、つまり周期は 113/6 ≈ 19 で、これは図2のピークの1つである周波数 k = 6 に対応します。活性は `a` と `b` のそれぞれについて同じ周波数で周期的に変わっており、加法定理で `cos(ω_k(a + b))` を計算するのに必要な入力になっています。

### 損失の精度と振動

現在のコード（公開実装に合わせた構造）で、損失の精度だけを変えて 40,000 ステップずつ学習し、比べました（CPU、`data_seed = init_seed = 0`、各1回）。

| 項目 | float32 | float64（現在のコード） |
|---|---|---|
| 訓練損失のスパイク | 30 回。約 1,300 ステップごと、最大 0.079 | なし |
| テスト精度が 0.5 / 0.99 を超えるステップ | 4,700 / 5,700 | 4,800 / 5,700 |
| 汎化した後のテスト精度の最小値 | 0.962 | 0.990 |
| 重みの二乗和（最終） | 1,119（1,000〜1,500 の間で上下し続ける） | 912 |
| `W_E` の上位の周波数 | 44, 6, 30, 16, 32 | 44, 16, 6, 20, 40 |

- 振動の原因は損失の精度で、モデルの構造ではありません。損失が 1e-7 前後まで下がると、float32 では多くの組で損失がちょうど 0 に丸められ、勾配の向きも正しい向きから大きくずれます。
- float32 では、スパイクとスパイクの間に重みの二乗和が増え、スパイクで急に下がる、のこぎり型の動きを繰り返します。丸めで偏った勾配を、Adam が学習率いっぱいの大きさの一歩に引き伸ばすためと考えられます（推論）。
- 振動があってもなくても、汎化する時期はほぼ同じでした。振動は grokking に必要なものでも、grokking を妨げるものでもありません。
- ただし float32 では、汎化した後もテスト精度が一時的に下がり、最後に使う周波数も変わります。上の図1〜3は float32 の版の結果なので、この影響を含んでいます。

詳しい数値は [`docs/experiments.md` の「損失の精度の比較実験」](docs/experiments.md#損失の精度の比較実験cpu40000-ステップ)にあります。

## ディレクトリ構成

```
grokking/            data.py, model.py, config.py, train.py
grokking/analysis/   fourier.py, neurons.py, plots.py
configs/             base.json, experiments/*.json
notebooks/           01_train.ipynb, 02_fourier_analysis.ipynb
tests/               pytest のテスト
figures/             README に載せる図
docs/                spec.md（仕様）、experiments.md（実験記録）
```

## 参考文献

Nanda, Chan, Lieberum, Smith, Steinhardt (2023). Progress measures for grokking via mechanistic interpretability. ICLR 2023. arXiv:2301.05217.

Power, Burda, Edwards, Babuschkin, Misra (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

Liu, Michaud, Tegmark (2023). Omnigrok: Grokking beyond algorithmic data. arXiv:2210.01117.

Loshchilov, Hutter (2019). Decoupled weight decay regularization. arXiv:1711.05101.

## ライセンス

`LICENSE` を参照してください。