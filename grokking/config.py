from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    p: int = 113
    train_frac: float = 0.3
    data_seed: int = 0          # 分割に使う。実験 C では init_seed だけを変える
    init_seed: int = 0          # 重みの初期値に使う
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    steps: int = 40000
    log_every: int = 100
    n_checkpoints: int = 40     # 対数間隔で保存する個数
    run_name: str = "base"

    def __post_init__(self):
        self.betas = tuple(self.betas)          # JSON のリストをタプルに戻す

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "Config":
        """JSON を読む。"base" キーがあれば、その設定を読んでから差分で上書きする。"""
        d = json.loads(Path(path).read_text())
        base = d.pop("base", None)
        cfg = cls.load(Path(path).parent / base) if base else cls()
        for k, v in {**d, **overrides}.items():
            if k not in {f.name for f in fields(cls)}:
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
        cfg.__post_init__()
        return cfg

    def save(self, path: str | Path, extra: dict | None = None):
        Path(path).write_text(json.dumps({**asdict(self), **(extra or {})}, indent=2))

    def checkpoint_steps(self) -> list[int]:
        """0 から steps までの対数間隔のステップ番号（重複を除く）。"""
        import numpy as np
        pts = np.unique(np.logspace(0, np.log10(self.steps), self.n_checkpoints).astype(int))
        return sorted({0, *pts.tolist(), self.steps})