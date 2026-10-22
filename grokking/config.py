from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    """設定の項目と型。値は configs/base.json にだけ書き、ここには既定値を持たせない。"""

    p: int
    train_frac: float
    data_seed: int              # 分割に使う。実験 C では init_seed だけを変える
    init_seed: int              # 重みの初期値
    d_model: int
    n_heads: int
    d_mlp: int
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    steps: int
    log_every: int
    n_checkpoints: int          # 対数間隔で保存する個数
    run_name: str

    def __post_init__(self):
        beta1, beta2 = self.betas
        self.betas = (beta1, beta2)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "Config":
        """JSON を読む。"base" キーがあれば、その設定を読んでから差分で上書きする。
        キーが足りなくても、知らないキーがあっても KeyError。
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        base = d.pop("base", None)
        d.pop("git", None)          # train.py が config.json に書く記録用のキー。設定ではない
        merged = asdict(cls.load(Path(path).parent / base)) if base else {}
        merged.update(d)
        merged.update(overrides)

        names = {f.name for f in fields(cls)}
        if unknown := merged.keys() - names:
            raise KeyError(f"unknown config key(s): {sorted(unknown)}")
        if missing := names - merged.keys():
            raise KeyError(f"missing config key(s): {sorted(missing)}")
        return cls(**merged)        # __post_init__ もここで呼ばれる

    def save(self, path: str | Path, extra: dict | None = None):
        Path(path).write_text(json.dumps({**asdict(self), **(extra or {})}, indent=2))

    def checkpoint_steps(self) -> list[int]:
        """0 から steps までの対数間隔のステップ番号（重複を除く）。"""
        import numpy as np
        pts = np.unique(np.logspace(0, np.log10(self.steps), self.n_checkpoints).astype(int))
        return sorted({0, *pts.tolist(), self.steps})