"""JSONL metrics writer for training runs.

One JSON object per line, one line per logged optimizer step (log_interval).
Machine-readable for downstream plotting/analysis.
"""

from __future__ import annotations

import json
import os


class MetricsWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def write(self, step, epoch, train_loss=None, val_loss=None, learning_rate=None,
              wquad_grad_norm=None, wlin_grad_norm=None, other_grad_norm=None,
              wquad_param_norm=None, wlin_param_norm=None,
              wquad_update_norm=None, wlin_update_norm=None,
              activations=None, **extra):
        row = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "wquad_grad_norm": wquad_grad_norm,
            "wlin_grad_norm": wlin_grad_norm,
            "other_grad_norm": other_grad_norm,
            "wquad_param_norm": wquad_param_norm,
            "wlin_param_norm": wlin_param_norm,
            "wquad_update_norm": wquad_update_norm,
            "wlin_update_norm": wlin_update_norm,
        }
        if activations:
            for k, v in activations.items():
                row[f"act_{k}"] = v
        row.update(extra)
        # JSON-serialize floats; numpy types are handled via float()
        row = {k: _serialize(v) for k, v in row.items() if v is not None}
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()
        self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serialize(v):
    if isinstance(v, (float, int, str, bool)) or v is None:
        return v
    try:
        import numpy as np
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    try:
        return float(v)  # torch tensors
    except Exception:
        return str(v)
