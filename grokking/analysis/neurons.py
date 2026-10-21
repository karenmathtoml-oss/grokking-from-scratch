from __future__ import annotations

import torch

from grokking.data import CyclicGroup, make_dataset


@torch.no_grad()
def capture(model, hook_module, tokens: torch.Tensor) -> torch.Tensor:
    """hook_module を通る値を forward で取り出す。"""
    store = {}

    def save(module, inputs, output):
        store.setdefault("x", output.detach())

    handle = hook_module.register_forward_hook(save)
    try:
        model(tokens)
    finally:
        handle.remove()
    return store["x"]


def mlp_activations(model) -> torch.Tensor:
    """全組 (a, b) に対する MLP 活性を (p, p, d_mlp) で返す。acts[a, b, n] がニューロン n の値。"""
    tokens, _ = make_dataset(CyclicGroup(model.p))
    device = next(model.parameters()).device
    acts = capture(model, model.hook_mlp_act, tokens.to(device))    # (p*p, d_mlp)
    acts = acts.view(model.p, model.p, -1)                          # 第1軸 a、第2軸 b
    return acts.cpu()