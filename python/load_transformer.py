"""Load the C++ flat weight buffer into the PyTorch replica.

Flat layout (verified against model/network.cpp + setupNeuralNetwork):

    embedding:                       [vocab * emb]                          -> (vocab, emb)
    per block (4x):
      norm1:                         [W_quad][W_lin][bias][gamma][beta]
      attention 1..4:                [Wq][Wk][Wv]                           (each emb*emb)
      norm2:                         [W_quad][W_lin][bias][gamma][beta]
      FF1 (emb->ff):                 [W_quad][W_lin][bias]
      FF2 (ff->emb):                 [W_quad][W_lin][bias]
    final_norm:                      [W_quad][W_lin][bias][gamma][beta]
    output (emb->vocab):             [W_quad][W_lin][bias]

Matrix convention (critical, verified from C++ matmult):
  - Dense quadratic weights are stored row-major (out, in) and applied as
    y = x @ W.T, so `W_pt = flat.reshape(out, in)`  (NO transpose).
  - Attention Q/K/V weights are stored row-major (emb, emb) and applied as
    x @ W (NO transpose), so `W_pt = flat.reshape(emb, emb)`.
"""

import os

import torch

from model import Transformer


def load_flat_weights(path):
    """Read a raw little-endian float32 array from disk."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"Weight file size {len(data)} not divisible by 4")
    return torch.frombuffer(bytearray(data), dtype=torch.float32).clone()


def take(weights, offset, count):
    end = offset + count
    if end > len(weights):
        raise ValueError(
            f"Ran past end of weight file: {end} > {len(weights)} (offset {offset}, count {count})"
        )
    return weights[offset:end], end


def load_quadratic_linear(dense, weights, offset, in_features, out_features):
    """[W_quad (out,in)][W_lin (out,in)][bias (out)] -> QuadraticLinear."""
    n = in_features * out_features
    wq, offset = take(weights, offset, n)
    dense.W_quad.data.copy_(wq.reshape(out_features, in_features))
    wl, offset = take(weights, offset, n)
    dense.W_lin.data.copy_(wl.reshape(out_features, in_features))
    bias, offset = take(weights, offset, out_features)
    dense.bias.data.copy_(bias)
    return offset


def load_norm(norm, weights, offset, dim):
    """[W_quad][W_lin][bias][gamma][beta] -> QuadNorm."""
    offset = load_quadratic_linear(norm.dense, weights, offset, dim, dim)
    gamma, offset = take(weights, offset, dim)
    beta, offset = take(weights, offset, dim)
    norm.ln.weight.data.copy_(gamma)
    norm.ln.bias.data.copy_(beta)
    return offset


def load_attention(head, weights, offset, dim):
    """[Wq][Wk][Wv], each (dim*dim) row-major, applied as x @ W."""
    n = dim * dim
    for param in (head.Wq, head.Wk, head.Wv):
        values, offset = take(weights, offset, n)
        param.data.copy_(values.reshape(dim, dim))
    return offset


def build_and_load(
    vocabulary_size,
    weights_path,
    embedding_dimension=128,
    feedforward_dimension=512,
    transformer_layers=4,
    attention_heads=4,
):
    """Build the replica and load the C++ flat weights.

    Returns (model, final_offset). Raises if the offset does not exactly match
    the number of floats in the file.
    """
    model = Transformer(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
        feedforward_dimension=feedforward_dimension,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
    )
    weights = load_flat_weights(weights_path)
    offset = 0

    # embedding: [vocab * emb] -> (vocab, emb)
    n = vocabulary_size * embedding_dimension
    emb_weights, offset = take(weights, offset, n)
    model.embedding.weight.data.copy_(
        emb_weights.reshape(vocabulary_size, embedding_dimension)
    )

    for b in range(transformer_layers):
        block = model.blocks[b]
        offset = load_norm(block.norm1, weights, offset, embedding_dimension)
        for h in range(attention_heads):
            offset = load_attention(block.attention[h], weights, offset, embedding_dimension)
        offset = load_norm(block.norm2, weights, offset, embedding_dimension)
        offset = load_quadratic_linear(
            block.ff1, weights, offset, embedding_dimension, feedforward_dimension
        )
        offset = load_quadratic_linear(
            block.ff2, weights, offset, feedforward_dimension, embedding_dimension
        )

    offset = load_norm(model.final_norm, weights, offset, embedding_dimension)
    offset = load_quadratic_linear(
        model.output, weights, offset, embedding_dimension, vocabulary_size
    )

    if offset != len(weights):
        raise ValueError(
            f"WEIGHT MISMATCH: consumed {offset} floats but file has {len(weights)} "
            f"({len(weights) - offset} unconsumed / {offset - len(weights)} over-consumed)"
        )
    return model, offset


if __name__ == "__main__":
    import sys

    vocab = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "ref_data", "fresh_transformer_weights.bin"
    )
    model, offset = build_and_load(vocab, path)
    print(f"Loaded {offset:,} floats.")
    print(f"Expected total: 2,185,216")
    print("SUCCESS" if offset == 2_185_216 else "WARNING: offset != 2,185,216")
