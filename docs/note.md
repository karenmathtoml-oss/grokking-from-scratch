# 実装メモ（詰まったところと判断の記録）

## 詰まったところ

| 症状 | 原因 | 対処 |
|---|---|---|
| `No module named pytest` | 仮想環境に pytest がない | `.venv` を作って有効化し、`python -m pip install pytest` |
| `python -m .\tests\test_data.py` が失敗 | `-m` にはファイルパスではなくモジュール名を渡す | `python -m pytest tests -q` で実行する |
| 分割のテストで個数が 1 ずれる | `1 - 0.8 = 0.19999999999999996` で、`int(49 × 0.2)` が 9 になる | テストの個数は「全体 − 訓練」で書く |
| `'list' object has no attribute 'tolist'` | `tokens.tolist()` の要素はすでにリスト | テンソルの段階で `tokens[:, :2]` を切ってから `.tolist()` |
| `No module named 'grokking.analysis'` | `analysis/__init__.py` と `neurons.py`（`capture`）がまだなかった | 空の `__init__.py` と `capture` を先に作る |
| パラメータ数のテストが 1568 で失敗 | テスト用の小さいモデルで数えていた | `Config()` の基準設定で作ったモデルで数える（226,048） |
| GIF の1コマが 0 ms になる | imageio 2.37.3 の `mimsave` は `duration` をミリ秒で受け取る | `duration=300`（ミリ秒）で渡す |
| `git add` で LF/CRLF の警告 | Windows の改行コードの自動変換 | 動作に影響しないので、そのまま進める |

## 型チェッカー（Pylance）の警告

| 警告 | 原因 | 対処 |
|---|---|---|
| `int` に `item` がない | `sum()` は整数 0 から足し始めるので、結果が `int` かもしれない | 各要素で `.item()` してから合計する |
| `tuple[float, ...]` を `tuple[float, float]` に割り当てられない | `tuple(リスト)` は要素数が型に現れない | `beta1, beta2 = self.betas` で取り出して組み直す |
| `Module` に `__getitem__` がない | `register_buffer` の名前は `nn.Module.__getattr__` 経由で `Tensor \| Module` になる | クラスの先頭で `mask: torch.Tensor` と宣言する |
| `Tensor \| Module` を `Tensor` に渡せない | 引数を `nn.Module` と型指定していた | `Callable[[], Transformer]` にする（`TYPE_CHECKING` で import） |

## 設計上の判断

| 判断 | 理由 |
|---|---|
| `Transformer` の既定値を消し、`Config` に一本化 | 引数の渡し忘れがエラーにならず、記録と実体がずれるのを防ぐ |
| `Config` の既定値も消し、値は `base.json` だけに書く | 同じ値が2か所にあると、片方だけ直したときにずれる。キーの書き忘れも既定値で黙って埋めずにエラーにする。後でキー（実験 F の `init_scale` など）を足したら、それ以前の `runs/*/config.json` は `Config.load(path, init_scale=1.0)` のように足りないキーを明示して読む |
| `seq_len` を `Config` に入れない | 系列長はデータ `[a, b, =]` が決める。食い違いを構造的に防ぐ |
| `config.json` に git ハッシュを書く | どのコードの結果かを辿れる。`Config.load` はこのキーを読み飛ばす |
| 出力先が空でなければ止める | 同じ `run_name` で再実行すると、古いチェックポイントが混ざる |
| `capture` の hook は値を返さない | forward hook は `None` 以外を返すと出力を置き換える。`lambda` で `setdefault` の値を返すと、2回目以降の出力が1回目の値に化ける |
| 解析の関数に `@torch.no_grad()` | `requires_grad` のテンソルは `.numpy()` できない。呼び出し側の `detach` 忘れでも壊れない |

## 実装ガイドのコードとの差分

CSV の列名は `train_accuracy` / `test_accuracy`、チェックポイントは `run_dir` 直下の `ckpt_*.pt` に `state_dict` を直接保存、`train.py` に `--resume` と `--steps` はない、`Transformer` は6引数すべてが必須、`capture` は `def` で書き値を返さない、`spectrum_gif` の `duration` はミリ秒。公開ノートブック（Nanda et al.、ARENA 1.5.2）との比較は、図が出てから行う。

## スモーク実行の記録（2,000 ステップ、CPU、242 秒）

| ステップ | train_loss | train_acc | test_loss | test_acc | weight_sq_norm |
|---|---|---|---|---|---|
| 0 | 4.7407 | 0.0086 | 4.7391 | 0.0092 | 1383.03 |
| 100 | 1.9716 | 0.5488 | 8.3485 | 0.0126 | 2512.74 |
| 2000 | 0.00028 | 1.0 | 16.85 | 0.0776 | 3154.10 |

ステップ 0 の損失は ln 113 ≈ 4.727、正解率は 1/113 ≈ 0.0088、二乗和は理論値 Σ(個数 / fan_in) = 1382 とほぼ一致した。2,000 ステップでは丸暗記の段階で、テスト損失は増え続けている。ファイル数は 38（チェックポイント 36 個）。

## 確認問題（答えは自分の言葉で）

| 工程 | 問い | 答え |
|---|---|---|
| 1 | `=` の ID を 0 ではなく p にするのはなぜか | |
| 1 | 分割がグローバル乱数に依存しないことは、実験 C で何を保証するか | |
| 1 | 訓練組 (a, b) のうち、(b, a) も訓練に入っている割合はいくらか。モデルに有利か | |
| 2 | `einsum("bsd,hde->bhse", x0, W_Q)` がエラーになるのはなぜか | |
| 2 | 位置 0 と 1 の MLP を計算しなくても出力が変わらないのはなぜか。2層ならどうか | |
| 2 | `mask` を buffer にする理由（weight decay とノルムの観点） | |
| 2 | パラメータ数を数えよ | |
| 3 | `betas` を `__post_init__` でタプルに戻す理由 | |
| 3 | 実験 B と F は、どのキーを足せば表せるか。`model.py` のどこを変えるか | |
| 4 | 対数間隔のチェックポイントは何を見るのに向いているか | |
| 4 | なぜフルバッチか。ミニバッチだと再現性と観察に何が変わるか | |
| 4 | `model.eval()` / `train()` で挙動が変わるのは、どんな層があるときか | |
| 5 | p が偶数だと基底は何本足りないか。k = p/2 の sin はどうなるか | |
| 5 | k を 1〜(p−1)/2 に限るのはなぜか | |
| 5 | 学習前（ステップ 0）のスペクトルはどんな形か | |
| 6 | 同じ `Identity` が2回呼ばれたら、`capture` はどちらの値を返すか | |
| 6 | `view(p, p, -1)` で軸が (a, b) になる根拠 | |
| 6 | 図3の縞の間隔が Δ のとき、周波数 k はいくらか | |

## 残りの作業

Colab（GPU）でスモーク実行の時間を測り、本番の所要時間を記録する。本番（`configs/base.json`）を回し、テスト精度が 0.5 を超えた最初のステップと最終行をここに書く。図1〜3を `figures/` に置いて README に載せる。公開ノートブックとの差分（ハイパーパラメータ、初期化のスケール、MLP のバイアス、Attention のマスク）を確かめる。