from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_curves(metrics_csv: str | Path, out: str | Path | None = None):
    """図1：損失と精度の対数グラフ"""
    df = pd.read_csv(metrics_csv)
    df = df[df.step > 0]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax_loss.plot(df.step, df.train_loss, label="train")
    ax_loss.plot(df.step, df.test_loss, label="test")
    ax_loss.set(xscale="log", yscale="log", xlabel="step", ylabel="loss")
    ax_loss.legend()

    ax_acc.plot(df.step, df.train_accuracy, label="train")
    ax_acc.plot(df.step, df.test_accuracy, label="test")
    ax_acc.set(xscale="log", xlabel="step", ylabel="accuracy")
    ax_acc.legend()

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def plot_spectrum(spec: torch.Tensor, title: str = "", ax=None):
    """図2：周波数 k ごとの強さの棒グラフ。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    k = range(1, len(spec) + 1)
    ax.bar(k, spec.detach().cpu().numpy())
    ax.set(xlabel="frequency k", ylabel="norm", title=title)
    return ax


def spectrum_gif(run_dir: str | Path, p: int, out: str | Path,
                 build_model: Callable[[], torch.nn.Module], frame_ms: int = 300) -> None:
    """チェックポイントごとの埋め込みスペクトルを GIF にする。

    build_model() は、空のモデルを返す関数。
    frame_ms は1コマの長さ（ミリ秒）。
    """
    import imageio.v2 as imageio

    from grokking.analysis.fourier import embedding_spectrum

    frames = []
    for ckpt in sorted(Path(run_dir).glob("ckpt_*.pt")):
        step = int(ckpt.stem.split("_")[1])                 # ckpt_040000 → 40000
        model = build_model()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        spec = embedding_spectrum(model.W_E, p)

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_spectrum(spec, title=f"step {step}", ax=ax)
        ax.set_ylim(0, None)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out, frames, duration=frame_ms)          # imageio 2.3x は ミリ秒単位


def plot_neuron(acts: torch.Tensor, n: int, out: str | Path | None = None):
    """図3：ニューロン n の活性のヒートマップ。acts は (p, p, d_mlp)。"""
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(acts[:, :, n].numpy(), origin="lower")
    ax.set(xlabel="b", ylabel="a", title=f"neuron {n}")
    fig.colorbar(image, ax=ax)
    if out:
        fig.savefig(out, dpi=150)
    return fig