import json
from pathlib import Path

import pytest

from grokking.config import Config


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "base.json"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_save_load_roundtrip(tmp_path):
    cfg = Config.load(BASE, p=7, weight_decay=0.1, run_name="rt")
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == cfg


def test_base_overrides_only_given_keys(tmp_path):
    Config.load(BASE, p=7, weight_decay=1.0).save(tmp_path / "base.json")
    exp = write_json(tmp_path / "experiments" / "A.json",
                     {"base": "../base.json", "weight_decay": 0.0, "run_name": "A"})
    cfg = Config.load(exp)
    assert cfg.weight_decay == 0.0          # 差分で上書きされた
    assert cfg.p == 7                       # base から引き継がれた
    assert cfg.lr == Config.load(BASE).lr   # どちらにも書かれていない値は、base の値のまま


def test_load_skips_git_key(tmp_path):
    """train.py が書く config.json（git ハッシュ付き）を読み直せる。"""
    cfg = Config.load(BASE, p=7, run_name="rt")
    cfg.save(tmp_path / "config.json", extra={"git": "0123abc"})
    assert Config.load(tmp_path / "config.json") == cfg


def test_unknown_key_raises_even_with_git(tmp_path):
    """git を読み飛ばしても、それ以外の知らないキーはエラーのまま。"""
    Config.load(BASE).save(tmp_path / "c.json", extra={"git": "0123abc", "weight_decya": 0.0})
    with pytest.raises(KeyError, match="weight_decya"):
        Config.load(tmp_path / "c.json")


def test_unknown_key_raises(tmp_path):
    Config.load(BASE).save(tmp_path / "c.json", extra={"weight_decya": 0.0})
    with pytest.raises(KeyError, match="weight_decya"):
        Config.load(tmp_path / "c.json")


def test_missing_key_raises(tmp_path):
    """既定値はないので、キーの書き忘れはエラーになる。"""
    d = json.loads(BASE.read_text(encoding="utf-8"))
    del d["lr"]
    path = write_json(tmp_path / "c.json", d)
    with pytest.raises(KeyError, match="lr"):
        Config.load(path)


def test_checkpoint_steps_properties():
    s = Config.load(BASE, steps=40000, n_checkpoints=40).checkpoint_steps()
    assert s[0] == 0 and s[-1] == 40000
    assert s == sorted(set(s))                      # 昇順・重複なし
    assert len(s) <= 40 + 2
    assert all(isinstance(x, int) for x in s)       # numpy の整数が混ざっていない
    assert s[1] - s[0] < s[-1] - s[-2]              # 対数間隔：序盤は密、終盤は疎
