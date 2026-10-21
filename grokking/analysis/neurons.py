"""forward hook で、モデルの中間値を取り出す。"""
from __future__ import annotations

import torch


@torch.no_grad()
def capture(model, hook_module, tokens: torch.Tensor) -> torch.Tensor:
    """hook_module を通る値を、1回の forward で取り出す。"""
    store = {}

    def save(module, inputs, output):
        store.setdefault("x", output.detach())

    handle = hook_module.register_forward_hook(save)
    try:
        model(tokens)
    finally:
        handle.remove()
    return store["x"]