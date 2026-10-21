import json
from pathlib import Path

import pytest

from grokking.config import Config


ROOT = Path(__file__).resolve().parent.parent


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_save_load_roundtrip(tmp_path):
    cfg = Config(p=7, weight_decay=0.1, run_name="rt")
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == cfg


def test_base_overrides_only_given_keys(tmp_path):
    write_json(tmp_path / "base.json", {"p": 7, "weight_decay": 1.0})
    exp = write_json(tmp_path / "experiments" / "A.json",
                     {"base": "../base.json", "weight_decay": 0.0, "run_name": "A"})
    cfg = Config.load(exp)
    assert cfg.weight_decay == 0.0      # 差分で上書きされた
    assert cfg.p == 7                   # base から引き継がれた
    assert cfg.lr == Config().lr        # どちらにも書かれていない値は、既定値のまま


def test_load_skips_git_key(tmp_path):
    """train.py が書く config.json（git ハッシュ付き）を読み直せる。"""
    cfg = Config(p=7, run_name="rt")
    cfg.save(tmp_path / "config.json", extra={"git": "0123abc"})
    assert Config.load(tmp_path / "config.json") == cfg


def test_unknown_key_raises_even_with_git(tmp_path):
    """git を読み飛ばしても、それ以外の知らないキーはエラーのまま。"""
    path = write_json(tmp_path / "c.json", {"git": "0123abc", "weight_decya": 0.0})
    with pytest.raises(KeyError):
        Config.load(path)


def test_unknown_key_raises(tmp_path):
    path = write_json(tmp_path / "c.json", {"weight_decya": 0.0})
    with pytest.raises(KeyError):
        Config.load(path)


def test_checkpoint_steps_properties():
    s = Config(steps=40000, n_checkpoints=40).checkpoint_steps()
    assert s[0] == 0 and s[-1] == 40000
    assert s == sorted(set(s))                      # 昇順・重複なし
    assert len(s) <= 40 + 2
    assert all(isinstance(x, int) for x in s)       # numpy の整数が混ざっていない
    assert s[1] - s[0] < s[-1] - s[-2]              # 対数間隔：序盤は密、終盤は疎


def test_base_json_matches_defaults():
    assert Config.load(ROOT / "configs" / "base.json") == Config()