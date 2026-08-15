"""Configuration-agnostic parameter enumeration and optimizer grouping.

The C++ flat weight layout is, per block: norm1 -> attention heads -> norm2 ->
ff1 -> ff2, with each dense layer laid out [W_quad][W_lin][bias] and each norm
adding [gamma][beta]. This module reproduces that enumeration order for ANY
number of blocks / heads / dimensions by walking the model structure instead
of hardcoding 4 heads or 4 blocks.

``iter_parameters(model)`` yields (name, param, lr_scale, wd_scale) in exact
C++ flat order. ``build_param_groups`` turns that into torch.optim param
groups with the quadratic 0.7 (or configured) LR/WD scaling.

The default quad scales (0.7) match the validated C++ train_adams
(quad_lr_scale = quad_wd_scale = 0.7); the audit scripts keep their own local
copies so they remain untouched.
"""

from __future__ import annotations

import torch


def _iter_dense(name, dense, quad_lr, quad_wd):
    yield f"{name}.W_quad", dense.W_quad, quad_lr, quad_wd
    yield f"{name}.W_lin", dense.W_lin, 1.0, 1.0
    yield f"{name}.bias", dense.bias, 1.0, 1.0


def _iter_norm(name, norm, quad_lr, quad_wd):
    yield from _iter_dense(name, norm.dense, quad_lr, quad_wd)
    yield f"{name}.gamma", norm.ln.weight, 1.0, 1.0
    yield f"{name}.beta", norm.ln.bias, 1.0, 1.0


def iter_parameters(model, quad_lr=0.7, quad_wd=0.7):
    """Yield (name, param, lr_scale, wd_scale) in C++ flat order.

    Derives block/head counts from the model itself, so it works for any
    config (1 or N blocks, 1 or N heads, any embedding/FF dimension).
    """
    yield "embedding.weight", model.embedding.weight, 1.0, 1.0
    for b, block in enumerate(model.blocks):
        yield from _iter_norm(f"block{b+1}_norm1", block.norm1, quad_lr, quad_wd)
        for h, head in enumerate(block.attention):
            yield f"block{b+1}_head{h+1}.Wq", head.Wq, 1.0, 1.0
            yield f"block{b+1}_head{h+1}.Wk", head.Wk, 1.0, 1.0
            yield f"block{b+1}_head{h+1}.Wv", head.Wv, 1.0, 1.0
        yield from _iter_norm(f"block{b+1}_norm2", block.norm2, quad_lr, quad_wd)
        yield from _iter_dense(f"block{b+1}_ff1", block.ff1, quad_lr, quad_wd)
        yield from _iter_dense(f"block{b+1}_ff2", block.ff2, quad_lr, quad_wd)
    yield from _iter_norm("final_norm", model.final_norm, quad_lr, quad_wd)
    yield from _iter_dense("output", model.output, quad_lr, quad_wd)


def class_of(name):
    """Parameter class for metrics (matches the audit scripts' classification)."""
    if name == "embedding.weight":
        return "embedding"
    if name.endswith(".W_quad"):
        return "W_quad"
    if name.endswith(".W_lin"):
        return "W_lin"
    if name.endswith(".bias"):
        return "bias"
    if name.endswith(".gamma") or name.endswith(".beta"):
        return "gamma_beta"
    return "attention"  # Wq/Wk/Wv


def build_param_groups(model, lr_eff, weight_decay, quad_scale, quadratic=True):
    """Return torch.optim parameter groups with C++-matching scales.

    Normal params:   lr = lr_eff,            wd = weight_decay
    W_quad params:   lr = quad_scale*lr_eff, wd = quad_scale*weight_decay
    (algebraically identical to C++ train_adams:
     p -= lr_eff*s*(adam + wd*d*p)  with s=d=quad_scale on W_quad blocks)

    If ``quadratic`` is False the W_quad parameters are omitted from the
    optimizer entirely (they are expected to have been zeroed separately).
    """
    groups = {"default": [], "quad": []}
    for name, param, lr_s, wd_s in iter_parameters(model, quad_lr=quad_scale, quad_wd=quad_scale):
        if not quadratic and name.endswith(".W_quad"):
            continue
        if name.endswith(".W_quad"):
            groups["quad"].append(param)
        else:
            groups["default"].append(param)
    param_groups = [
        {"params": groups["default"], "lr": lr_eff, "weight_decay": weight_decay},
    ]
    if groups["quad"]:
        param_groups.append({
            "params": groups["quad"],
            "lr": quad_scale * lr_eff,
            "weight_decay": quad_scale * weight_decay,
        })
    return param_groups


def zero_w_quad(model):
    """Zero every W_quad parameter in place (quadratic=False semantics)."""
    with torch.no_grad():
        for name, param, _, _ in iter_parameters(model):
            if name.endswith(".W_quad"):
                param.zero_()
