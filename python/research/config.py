"""Research configuration for the quadratic transformer.

Defaults match the validated C++ reference (documentation/example_transformer.cpp
and the Stage 7-9 equivalence tests) wherever a C++ default exists:
    emb=128, ff=512, 4 blocks, 4 sequential single-head attentions,
    max_seq=128, batch_size=8, lr=0.001, min_lr=1e-5, betas=(0.9, 0.999),
    eps=1e-8, weight_decay=0.01, quad_scale=0.7, window stride = max_seq/2,
    no shuffle, cosine schedule reset per epoch, global step for Adam.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- reproducibility ---
    seed: int = 1
    device: str = "cpu"

    # --- model ---
    vocab_size: int = 243            # TinyStories char vocab (243); override after tokenizing
    embedding_dimension: int = 128
    feedforward_dimension: int = 512
    transformer_layers: int = 4
    attention_heads: int = 4
    max_sequence_length: int = 128

    # --- data ---
    train_path: str = "data/TinyStories/tinystories-train.txt"
    valid_path: str = "data/TinyStories/tinystories-valid.txt"
    batch_size: int = 8
    window_stride: int = 0            # 0 -> max_sequence_length // 2 (C++ stride)
    shuffle: bool = False             # C++ never shuffles

    # --- optimizer (C++ train_adams semantics) ---
    learning_rate: float = 0.001      # divided by batch_size inside the trainer (lr_eff)
    min_learning_rate: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    quad_scale: float = 0.7           # lr and wd scale for W_quad blocks (quad_lr_scale)
    quadratic: bool = True            # False -> zero W_quad and exclude from optimizer

    # --- training ---
    epochs: int = 1
    epochs_per_checkpoint: int = 1
    log_interval: int = 1             # metric rows per optimizer step

    # --- outputs ---
    output_dir: str = "runs/default"

    def __post_init__(self):
        if self.window_stride <= 0:
            self.window_stride = self.max_sequence_length // 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.__post_init__()
        return cfg

    def resolve_paths(self, base=None):
        """Make relative paths absolute relative to `base` (default: repo root)."""
        base = base or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return {
            "train_path": os.path.join(base, self.train_path) if not os.path.isabs(self.train_path) else self.train_path,
            "valid_path": os.path.join(base, self.valid_path) if not os.path.isabs(self.valid_path) else self.valid_path,
            "output_dir": os.path.join(base, self.output_dir) if not os.path.isabs(self.output_dir) else self.output_dir,
        }

    def __repr__(self):
        lines = [f"Config({self.__class__.__name__})"]
        for k, v in asdict(self).items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)
