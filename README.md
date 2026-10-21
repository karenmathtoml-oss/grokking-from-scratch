# grokking-from-scratch

English | [日本語](README.ja.md)

A from-scratch reproduction of *Progress measures for grokking via mechanistic interpretability* (Nanda et al., 2023). A one-layer Transformer learns addition modulo 113 from 30% of all pairs. It memorizes the training set within a few hundred steps and generalizes to the held-out pairs only much later ("grokking"). The repository also contains the Fourier analysis used to show that the trained network computes `a + b` with a handful of frequencies.

**Status:** all code (data, model, config, training loop, Fourier analysis, plots) is implemented and covered by tests. The full 40,000-step run and the final figures are pending.

## Setup

Run every command from the repository root.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install torch numpy pandas matplotlib imageio pytest
python -m pytest tests -q
```

## Training

```bash
python -m grokking.train --config configs/base.json
```

The full run is intended for Google Colab with a GPU (`notebooks/01_train.ipynb`). On a CPU, a 2,000-step smoke run (`configs/experiments/smoke.json`) took 242 s, i.e. about 0.12 s per step.

Each run writes to `runs/<run_name>/`, which is not tracked by git.

| File | Content |
|---|---|
| `config.json` | All settings plus the git commit hash of the code |
| `metrics.csv` | `step, train_loss, train_accuracy, test_loss, test_accuracy, weight_sq_norm`, every `log_every` steps |
| `ckpt_XXXXXX.pt` | Model `state_dict` at log-spaced steps (about 40 per run) |

## Configuration

`configs/base.json` holds the settings of the paper. `Config` in `grokking/config.py` is the single source of default values, and a test checks that the two agree.

| Key | Value | Key | Value |
|---|---|---|---|
| `p` | 113 | `lr` | 1e-3 |
| `train_frac` | 0.3 | `weight_decay` | 1.0 |
| `d_model` | 128 | `betas` | (0.9, 0.98) |
| `n_heads` | 4 | `steps` | 40,000 (full batch, AdamW) |
| `d_mlp` | 512 | `log_every` / `n_checkpoints` | 100 / 40 |

`data_seed` fixes the train/test split and `init_seed` fixes the initial weights; they are independent of each other. An experiment is a small JSON file that overrides the base settings, and unknown keys raise an error:

```json
{"base": "../base.json", "weight_decay": 0.0, "run_name": "A_wd_0"}
```

## Analysis and figures

`notebooks/02_fourier_analysis.ipynb` loads a checkpoint, computes the Fourier spectra of the embedding `W_E` and of the neuron-to-logit map `W_U W_out`, and produces the figures with `grokking/analysis/plots.py`.

| Figure | Content |
|---|---|
| `figures/fig1_curve.png` | Loss and accuracy curves (log-scaled steps) |
| `figures/fig2_spectrum.gif` | Evolution of the embedding spectrum across checkpoints |
| `figures/fig3_neuron.png` | Activation of one MLP neuron over all `(a, b)` |

Figures will be added after the full run.

<!-- ![Figure 1](figures/fig1_curve.png) -->
<!-- ![Figure 2](figures/fig2_spectrum.gif) -->
<!-- ![Figure 3](figures/fig3_neuron.png) -->

## Repository layout

```
grokking/            data.py, model.py, config.py, train.py
grokking/analysis/   fourier.py, neurons.py, plots.py
configs/             base.json, experiments/*.json
notebooks/           01_train.ipynb, 02_fourier_analysis.ipynb (outputs cleared)
tests/               pytest suite
docs/                spec.md (specification), notes.md (issues and lessons)
```

## References

Nanda, Chan, Lieberum, Smith, Steinhardt (2023). Progress measures for grokking via mechanistic interpretability. ICLR 2023. arXiv:2301.05217.

Power, Burda, Edwards, Babuschkin, Misra (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

Liu, Michaud, Tegmark (2023). Omnigrok: Grokking beyond algorithmic data. arXiv:2210.01117.

Loshchilov, Hutter (2019). Decoupled weight decay regularization. arXiv:1711.05101.

## License

See `LICENSE`.