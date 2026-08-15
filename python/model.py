"""Faithful PyTorch replica of the C++ quadratic transformer.

Reproduces documentation/example_transformer.cpp exactly:
  Embedding
  -> 4 blocks:
       QuadNorm (QuadraticLinear -> LayerNorm)
       -> 4 sequential single-head attention layers (NOT multi-head)
       -> QuadNorm
       -> QuadraticLinear(128 -> 512)
       -> ReLU
       -> QuadraticLinear(512 -> 128)
  -> final QuadNorm
  -> QuadraticLinear(128 -> vocab)

Key C++ conventions that MUST be preserved:
  - Every dense layer:  y = x @ W_lin.T + (x*x) @ W_quad.T + bias
  - Flat weight layout per dense layer: [W_quad (out,in)][W_lin (out,in)][bias (out)]
  - Attention: Q/K/V linear projections with NO bias, applied as x @ W (not x @ W.T),
    scale = 1/sqrt(dim), causal mask = -1e30, single head, no residual anywhere.
  - LayerNorm: per-token over features, biased variance, eps=1e-5.
"""

import math

import torch
import torch.nn as nn


class QuadraticLinear(nn.Module):
    """y = x @ W_lin.T + (x*x) @ W_quad.T + bias  (matches C++ Layer::forward)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_quad = nn.Parameter(torch.empty(out_features, in_features))
        self.W_lin = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        linear = x @ self.W_lin.t()
        quadratic = (x * x) @ self.W_quad.t()
        return linear + quadratic + self.bias


class QuadNorm(nn.Module):
    """QuadraticLinear -> LayerNorm (C++ norm block: dense layer whose output is normalized)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dense = QuadraticLinear(dim, dim)
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.ln(self.dense(x))


class SingleHeadAttention(nn.Module):
    """C++ AttentionForward: single-head causal attention, Q/K/V = x @ W (no bias)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.Wq = nn.Parameter(torch.empty(dim, dim))
        self.Wk = nn.Parameter(torch.empty(dim, dim))
        self.Wv = nn.Parameter(torch.empty(dim, dim))

    def forward(self, x):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dim))
        seq_len = x.shape[-2]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal, -1e30)
        attn = torch.softmax(scores, dim=-1)
        return attn @ v


class TransformerBlock(nn.Module):
    """One C++ transformer layer: QuadNorm -> 4x single-head attention -> QuadNorm -> FF1(ReLU) -> FF2."""

    def __init__(self, embedding_dimension, feedforward_dimension, heads):
        super().__init__()
        self.norm1 = QuadNorm(embedding_dimension)
        self.attention = nn.ModuleList(
            [SingleHeadAttention(embedding_dimension) for _ in range(heads)]
        )
        self.norm2 = QuadNorm(embedding_dimension)
        self.ff1 = QuadraticLinear(embedding_dimension, feedforward_dimension)
        self.ff2 = QuadraticLinear(feedforward_dimension, embedding_dimension)

    def forward(self, x):
        x = self.norm1(x)
        for head in self.attention:
            x = head(x)
        x = self.norm2(x)
        x = self.ff1(x)
        x = torch.relu(x)
        x = self.ff2(x)
        return x


class Transformer(nn.Module):
    """Full C++ quadratic transformer replica.

    Config defaults match documentation/example_transformer.cpp:
    emb=128, ff=512, 4 blocks, 4 sequential single-head attentions per block.
    """

    def forward_packed(self, tokens, lengths):
        """Run the exact C++ forward path on a padded (B, L) token grid with a
        validity mask, returning logits on the padded grid.

        ``tokens``: int64 (B, L) padded grid; ``lengths``: sequence lengths.
        Rows >= lengths[s] are padded and masked out of attention (softmax
        over exactly the valid key columns, like the C++ compacted layout).
        The padded positions in the returned logits are meaningless; extract
        valid rows with the mask when comparing to C++ packed output.

        Uses the Stage-9 validated padded/masked attention (see
        compare_varseq.build_forward_padded) without touching ``forward()``.
        """
        B, L = tokens.shape
        lens = torch.as_tensor(lengths, dtype=torch.long, device=tokens.device)
        mask = torch.arange(L, device=tokens.device)[None, :] < lens[:, None]  # (B, L)
        out = {}
        h = self.embedding(tokens)
        out["embedding"] = h
        for bi, block in enumerate(self.blocks):
            d = block.norm1.dense(h)
            out[f"block{bi+1}_norm1_dense"] = d
            h = block.norm1.ln(d)
            out[f"block{bi+1}_norm1"] = h
            for hi, head in enumerate(block.attention):
                q = h @ head.Wq
                k = h @ head.Wk
                v = h @ head.Wv
                out[f"block{bi+1}_head{hi+1}_q"] = q
                out[f"block{bi+1}_head{hi+1}_k"] = k
                out[f"block{bi+1}_head{hi+1}_v"] = v
                scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head.dim))
                causal = torch.triu(torch.ones(L, L, device=h.device), diagonal=1).bool()
                scores = scores.masked_fill(causal, -1e30)
                scores = scores.masked_fill((~mask).unsqueeze(1), -1e30)  # padded keys
                attn = torch.softmax(scores, dim=-1)
                out[f"block{bi+1}_head{hi+1}_scores"] = attn
                h = attn @ v
                out[f"block{bi+1}_head{hi+1}_out"] = h
            d = block.norm2.dense(h)
            out[f"block{bi+1}_norm2_dense"] = d
            h = block.norm2.ln(d)
            out[f"block{bi+1}_norm2"] = h
            d = block.ff1(h)
            out[f"block{bi+1}_ff1_dense"] = d
            h = torch.relu(d)
            out[f"block{bi+1}_ff1_relu"] = h
            h = block.ff2(h)
            out[f"block{bi+1}_ff2"] = h
        d = self.final_norm.dense(h)
        out["final_norm_dense"] = d
        h = self.final_norm.ln(d)
        out["final_norm"] = h
        out["logits"] = self.output(h)
        out["probs"] = torch.softmax(out["logits"], dim=-1)
        return out, mask

    def forward(self, x):
        """Return logits (the C++ output layer applies softmax only as a hook)."""
        return self.forward_debug(x)["logits"]

    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=128,
        feedforward_dimension=512,
        transformer_layers=4,
        attention_heads=4,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension, feedforward_dimension, attention_heads
                )
                for _ in range(transformer_layers)
            ]
        )
        self.final_norm = QuadNorm(embedding_dimension)
        self.output = QuadraticLinear(embedding_dimension, vocabulary_size)

    def forward(self, x):
        """Return logits (the C++ output layer applies softmax only as a hook)."""
        return self.forward_debug(x)["logits"]

    def forward_debug(self, x):
        """Run the exact forward path and return every intermediate tensor.

        Names match the C++ reference dumps written by python/export_reference.cpp:
          - *_dense  : pre-activation of a norm block (pre-LayerNorm) == C++ 'preact'
          - *_relu   : post-ReLU of FF1
          - *_scores : post-softmax attention probabilities (C++ scratch holds these)
          - logits   : pre-softmax output (C++ 'preact' of the output layer)
          - probs    : softmax(logits)
        """
        out = {}
        out["embedding"] = self.embedding(x)
        h = out["embedding"]
        for bi, block in enumerate(self.blocks):
            out[f"block{bi+1}_norm1_dense"] = block.norm1.dense(h)
            out[f"block{bi+1}_norm1"] = block.norm1.ln(out[f"block{bi+1}_norm1_dense"])
            h = out[f"block{bi+1}_norm1"]
            for hi, head in enumerate(block.attention):
                out[f"block{bi+1}_head{hi+1}_q"] = h @ head.Wq
                out[f"block{bi+1}_head{hi+1}_k"] = h @ head.Wk
                out[f"block{bi+1}_head{hi+1}_v"] = h @ head.Wv
                scores = (out[f"block{bi+1}_head{hi+1}_q"] @ out[
                    f"block{bi+1}_head{hi+1}_k"
                ].transpose(-2, -1)) * (1.0 / math.sqrt(head.dim))
                causal = torch.triu(
                    torch.ones(h.shape[1], h.shape[1], device=h.device), diagonal=1
                ).bool()
                scores = scores.masked_fill(causal, -1e30)
                attn = torch.softmax(scores, dim=-1)
                out[f"block{bi+1}_head{hi+1}_scores"] = attn
                out[f"block{bi+1}_head{hi+1}_out"] = attn @ out[
                    f"block{bi+1}_head{hi+1}_v"
                ]
                h = out[f"block{bi+1}_head{hi+1}_out"]
            out[f"block{bi+1}_norm2_dense"] = block.norm2.dense(h)
            out[f"block{bi+1}_norm2"] = block.norm2.ln(out[f"block{bi+1}_norm2_dense"])
            h = out[f"block{bi+1}_norm2"]
            out[f"block{bi+1}_ff1_dense"] = block.ff1(h)
            out[f"block{bi+1}_ff1_relu"] = torch.relu(out[f"block{bi+1}_ff1_dense"])
            h = out[f"block{bi+1}_ff1_relu"]
            out[f"block{bi+1}_ff2"] = block.ff2(h)
            h = out[f"block{bi+1}_ff2"]
        out["final_norm_dense"] = self.final_norm.dense(h)
        out["final_norm"] = self.final_norm.ln(out["final_norm_dense"])
        h = out["final_norm"]
        out["logits"] = self.output(h)
        out["probs"] = torch.softmax(out["logits"], dim=-1)
        return out

    def init_like_cpp(self, seed=1):
        """Replicate C++ xavier_initialisation (U(-b, b), b = sqrt(6/(in+out)),
        quadratic block scaled by 1/sqrt(3)) for cases where no weight file exists."""
        g = torch.Generator().manual_seed(seed)

        def u_(param, bound):
            with torch.no_grad():
                # use the seeded local generator, NOT the global RNG, so the
                # seed argument fully determines the initialization
                param.uniform_(-bound, bound, generator=g)

        # embedding: in = 1, out = emb
        b = math.sqrt(6.0 / (1 + self.embedding_dimension))
        u_(self.embedding.weight, b)

        def init_quad(dense, in_, out_):
            base = math.sqrt(6.0 / (in_ + out_))
            quad_b = base / math.sqrt(3.0)
            u_(dense.W_quad, quad_b)
            u_(dense.W_lin, base)
            u_(dense.bias, base)

        def init_norm(norm, dim):
            init_quad(norm.dense, dim, dim)
            base = math.sqrt(6.0 / (dim + dim))
            u_(norm.ln.weight, base)
            u_(norm.ln.bias, base)

        for block in self.blocks:
            init_norm(block.norm1, self.embedding_dimension)
            for head in block.attention:
                base = math.sqrt(6.0 / (2 * self.embedding_dimension))
                u_(head.Wq, base)
                u_(head.Wk, base)
                u_(head.Wv, base)
            init_norm(block.norm2, self.embedding_dimension)
            init_quad(block.ff1, self.embedding_dimension, block.ff1.out_features)
            init_quad(block.ff2, block.ff2.in_features, self.embedding_dimension)
        init_norm(self.final_norm, self.embedding_dimension)
        init_quad(self.output, self.embedding_dimension, self.output.out_features)
