"""Backward-compatible re-export of the faithful replica (see model.py)."""

from model import (
    QuadraticLinear,
    QuadNorm,
    SingleHeadAttention,
    Transformer,
    TransformerBlock,
)

# Legacy name kept for compatibility with older scripts.
QuadraticTransformer = Transformer

__all__ = [
    "QuadraticLinear",
    "QuadNorm",
    "SingleHeadAttention",
    "TransformerBlock",
    "Transformer",
    "QuadraticTransformer",
]
