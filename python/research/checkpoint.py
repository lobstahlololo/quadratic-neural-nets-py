"""Checkpointing: model + optimizer + scheduler + all RNG state.

Resume is deterministic: restoring model/optimizer/scheduler state and the
Python, NumPy and PyTorch RNG states reproduces the exact next step, provided
the training loop only draws randomness from those three sources.
"""

from __future__ import annotations

import os
import pickle
import random

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, global_step, epoch,
                    config_dict=None, extra=None):
    """Write a full checkpoint to ``path``."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": {d: torch.cuda.get_rng_state(d)
                           for d in range(torch.cuda.device_count())}
        if torch.cuda.is_available() else {},
        "config": config_dict,
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, device="cpu"):
    """Restore state from ``path``; returns (global_step, epoch, config, extra)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    for d, state in payload.get("torch_cuda_rng", {}).items():
        torch.cuda.set_rng_state(state, d)
    return (payload["global_step"], payload["epoch"], payload.get("config"),
            {k: v for k, v in payload.items()
             if k not in ("model", "optimizer", "scheduler", "global_step",
                          "epoch", "python_rng", "numpy_rng", "torch_rng",
                          "torch_cuda_rng", "config")})
