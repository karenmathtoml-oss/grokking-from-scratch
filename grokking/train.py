from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.config import Config
from grokking.data import CyclicGroup, make_dataset, train_test_split
from grokking.model import Transformer


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """float64 で計算する交差エントロピー。float32 だと損失が約 1.2e-7 の倍数に丸められ、スパイクの原因になる。"""
    return F.cross_entropy(logits.to(torch.float64), labels)


def warmup_factor(step: int, warmup_steps: int) -> float:
    """学習率に掛ける係数。公開実装と同じく、step 0 の更新は 0 倍から始まる。"""
    if warmup_steps == 0:
        return 1.0
    return min(step / warmup_steps, 1.0)


@torch.no_grad()
def evaluate(model: Transformer, tokens: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """(損失, 正解率) を返す。"""
    logits = model(tokens)
    loss = cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss.item(), accuracy.item()

def git_hash() -> str:
    """現在のコミットハッシュ。取得できなければ "unknown"。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(cfg: Config, root: str | Path = "runs", device: str | None = None) -> Path:
    """cfg に従って学習し、出力先フォルダを返す。"""
    run_dir = Path(root) / cfg.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} は既に存在します。run_name を変えるか、フォルダを削除してください")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json", extra={"git": git_hash()})
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokens, labels = make_dataset(CyclicGroup(cfg.p))
    train_tokens, train_labels, test_tokens, test_labels = train_test_split(
        tokens, labels, cfg.train_frac, cfg.data_seed
    )
    train_tokens, train_labels, test_tokens, test_labels = (
        tensor.to(device)
        for tensor in (train_tokens, train_labels, test_tokens, test_labels)
    )

    model = Transformer(
        p=cfg.p,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        seq_len=tokens.shape[1],
        init_seed=cfg.init_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_factor(step, cfg.warmup_steps)
    )

    log_steps = set(range(0, cfg.steps + 1, cfg.log_every))
    checkpoint_steps = set(cfg.checkpoint_steps())
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file:
        fieldnames = [
            "step",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "weight_sq_norm",
        ]
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(cfg.steps + 1):
            if step in log_steps or step == cfg.steps:
                model.eval()
                train_loss, train_accuracy = evaluate(model, train_tokens, train_labels)
                test_loss, test_accuracy = evaluate(model, test_tokens, test_labels)
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "weight_sq_norm": model.weight_sq_norm(),
                    }
                )
                metrics_file.flush()

            if step in checkpoint_steps:
                torch.save(model.state_dict(), run_dir / f"ckpt_{step:06d}.pt")

            if step == cfg.steps:
                break

            model.train()
            logits = model(train_tokens)
            loss = cross_entropy(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    args = parser.parse_args()
    train(Config.load(args.config))


if __name__ == "__main__":
    main()