"""Research optimizer: torch.optim.AdamW with C++ train_adams semantics.

The validated C++ update (boilerplate/train_functions.h, train_adams) is:

    lr_eff  = learning_rate / batch_size
    m       = beta1*m + (1-beta1)*g            (beta1 = 0.9)
    v       = beta2*v + (1-beta2)*g*g          (beta2 = 0.999)
    m_hat   = m / (1 - beta1^t)                (t = global step, 1-based)
    v_hat   = v / (1 - beta2^t)
    adam    = m_hat / (sqrt(v_hat) + eps)      (eps = 1e-8)
    p      -= lr_eff * s * (adam + wd * d * p) (decoupled weight decay)

with s = d = quad_scale (=0.7) on W_quad blocks and s = d = 1.0 elsewhere.

torch.optim.AdamW with explicit param groups reproduces this algebraically:
a group with lr = lr_eff*s and weight_decay = wd*d gives

    p -= (lr_eff*s) * (adam + (wd*d) * p)      == the C++ update.

Gradients must be SUM-reduced (reduction='sum' cross entropy) so the C++
per-token-summed gradient semantics are preserved; the lr/batch_size division
happens here via lr_eff. The 1e-8 epsilon in AdamW applies the same
``sqrt(v_hat) + eps`` formula.
"""

from __future__ import annotations

import torch

from .params import build_param_groups, zero_w_quad


def make_optimizer(model, lr_eff, weight_decay=0.01, quad_scale=0.7,
                   beta1=0.9, beta2=0.999, epsilon=1e-8, quadratic=True):
    """Create the AdamW optimizer with explicit C++-matching param groups.

    Args:
        lr_eff: learning_rate / batch_size (the C++ lr_eff).
        weight_decay: global weight decay (C++ 0.01).
        quad_scale: LR and WD scale for W_quad blocks (C++ 0.7).
        quadratic: if False, W_quad params are excluded from the optimizer
            (caller should zero_w_quad first).
    """
    groups = build_param_groups(model, lr_eff, weight_decay, quad_scale, quadratic=quadratic)
    return torch.optim.AdamW(
        groups,
        lr=lr_eff,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )
