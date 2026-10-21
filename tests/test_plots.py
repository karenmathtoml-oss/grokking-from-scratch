import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from grokking.analysis.plots import plot_curves, plot_neuron, spectrum_gif
from grokking.model import Transformer


p = 7


def build_model() -> Transformer:
    return Transformer(p, d_model=16, n_heads=2, d_mlp=8, seq_len=3, init_seed=0)


def test_plot_curves_writes_png(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    metrics = pd.DataFrame({
        "step": [0, 10, 100],
        "train_loss": [4.7, 1.0, 0.01],
        "train_accuracy": [0.01, 0.5, 1.0],
        "test_loss": [4.7, 5.0, 6.0],
        "test_accuracy": [0.01, 0.02, 0.05],
        "weight_sq_norm": [1.0, 2.0, 3.0],
    })
    metrics.to_csv(csv_path, index=False)
    fig = plot_curves(csv_path, tmp_path / "fig1.png")
    plt.close(fig)
    assert (tmp_path / "fig1.png").stat().st_size > 0


def test_plot_neuron_writes_png(tmp_path):
    fig = plot_neuron(torch.rand(p, p, 8), 3, tmp_path / "fig3.png")
    plt.close(fig)
    assert (tmp_path / "fig3.png").stat().st_size > 0


def test_spectrum_gif_writes_file(tmp_path):
    for step in (0, 5):
        torch.save(build_model().state_dict(), tmp_path / f"ckpt_{step:06d}.pt")
    spectrum_gif(tmp_path, p, tmp_path / "fig2.gif", build_model)
    assert (tmp_path / "fig2.gif").stat().st_size > 0