"""Stage-9 variable-length batch comparison (C++ robust_harness vs PyTorch replica).

The C++ reference stores variable-length sequences COMPACTED (packed back-to-back
by their actual lengths). The PyTorch side reconstructs the batch as a padded
(B, max_len) grid with a validity mask: attention masks causal future columns AND
padded key columns (softmax over exactly the columns C++ sees). Valid rows are
extracted back into packed order for comparison.

Compares, for arbitrary config (full or toy) dumped by python/robust_harness.cpp
mode=fwd:
  - per-layer outputs + preactivations (all 35 layers)
  - reported loss (mean over rows of -log(p_target + 1e-7))
  - per-layer weight gradients and per-layer input gradients
  - one train_adams step (post-step weights + moments)

Usage: python3 compare_varseq.py <ref_dir>
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from compare_forward import load_floats, load_ints, parse_meta
from load_transformer import build_and_load

torch.set_num_threads(1)

ONE_MINUS_7 = np.float32(1e-7)


def build_forward_padded(model, tok, lengths):
    """Run the exact C++ forward path on a padded (B,L) token grid.

    Returns (checkpoints dict of (B,L,F) tensors, mask (B,L)).
    mask[s, r] is True for valid positions r < lengths[s].
    """
    B, L = tok.shape
    lens = torch.tensor(lengths, dtype=torch.long, device=tok.device)
    mask = torch.arange(L, device=tok.device)[None, :] < lens[:, None]  # (B, L)

    out = {}
    h = model.embedding(tok)
    out["embedding"] = h
    for bi, block in enumerate(model.blocks):
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
    d = model.final_norm.dense(h)
    out["final_norm_dense"] = d
    h = model.final_norm.ln(d)
    out["final_norm"] = h
    out["logits"] = model.output(h)
    out["probs"] = torch.softmax(out["logits"], dim=-1)
    return out, mask


def packed_rows(t, mask):
    """Extract valid rows of a (B,L,...) tensor in C++ packed order."""
    idx = mask.nonzero(as_tuple=False)  # (total_rows, 2), row-major
    return t[idx[:, 0], idx[:, 1]]


def layer_kinds(blocks, heads):
    """Ordered list of (layer_idx, kind) mirroring the C++ architecture."""
    kinds = ["embedding"]
    for b in range(blocks):
        kinds.append("norm1")
        for h in range(heads):
            kinds.append("head")
        kinds.append("norm2")
        kinds.append("ff1")
        kinds.append("ff2")
    kinds.append("final_norm")
    kinds.append("output")
    return kinds


def block_pos(layer_idx, heads):
    """Map a flat C++ layer index to (block, within-block-position, head).

    Position 0=norm1, 1..heads=head, heads+1=norm2, heads+2=ff1, heads+3=ff2.
    """
    if layer_idx == 0:
        return 0, -1, 0
    # Layers per block: norm1 + heads + norm2 + ff1 + ff2 = 4 + heads.
    stride = 4 + heads
    b = (layer_idx - 1) // stride
    w = (layer_idx - 1) % stride
    if w == 0:
        return b, 0, 0
    if 1 <= w <= heads:
        return b, 1, w - 1
    return b, w - heads, 0


def torch_checkpoint(model, kind, bi, hi, out):
    """Return the torch (B,L,F) tensor matching a C++ layer of the given kind."""
    if kind == "embedding":
        return out["embedding"]
    if kind == "norm1":
        return out[f"block{bi+1}_norm1"]
    if kind == "head":
        return out[f"block{bi+1}_head{hi+1}_out"]
    if kind == "norm2":
        return out[f"block{bi+1}_norm2"]
    if kind == "ff1":
        return out[f"block{bi+1}_ff1_relu"]
    if kind == "ff2":
        return out[f"block{bi+1}_ff2"]
    if kind == "final_norm":
        return out["final_norm"]
    if kind == "output":
        return out["probs"]
    raise ValueError(kind)


def torch_preact(model, kind, bi, hi, out):
    """Return the torch pre-hook tensor for a layer with hooks (or None)."""
    if kind == "norm1":
        return out[f"block{bi+1}_norm1_dense"]
    if kind == "norm2":
        return out[f"block{bi+1}_norm2_dense"]
    if kind == "ff1":
        return out[f"block{bi+1}_ff1_dense"]
    if kind == "final_norm":
        return out["final_norm_dense"]
    if kind == "output":
        return out["logits"]
    return None


def torch_grad_flat(model, kind, bi, hi):
    """Flattened per-layer torch gradients in C++ flat order: [W_quad][W_lin][bias]
    for dense, [Wq][Wk][Wv] for heads, [weight] for embedding, plus extra [gamma][beta]."""
    if kind == "embedding":
        return model.embedding.weight.grad.detach().reshape(-1), None
    if kind in ("norm1", "norm2", "final_norm"):
        block = model.blocks[bi] if kind != "final_norm" else None
        dense = block.norm1.dense if kind == "norm1" else block.norm2.dense if kind == "norm2" else model.final_norm.dense
        ln = block.norm1.ln if kind == "norm1" else block.norm2.ln if kind == "norm2" else model.final_norm.ln
        main = torch.cat([dense.W_quad.grad.detach().reshape(-1),
                          dense.W_lin.grad.detach().reshape(-1),
                          dense.bias.grad.detach().reshape(-1)])
        extra = torch.cat([ln.weight.grad.detach().reshape(-1),
                           ln.bias.grad.detach().reshape(-1)])
        return main, extra
    if kind == "head":
        head = model.blocks[bi].attention[hi]
        return torch.cat([head.Wq.grad.detach().reshape(-1),
                          head.Wk.grad.detach().reshape(-1),
                          head.Wv.grad.detach().reshape(-1)]), None
    if kind in ("ff1", "ff2"):
        dense = model.blocks[bi].ff1 if kind == "ff1" else model.blocks[bi].ff2
        return torch.cat([dense.W_quad.grad.detach().reshape(-1),
                          dense.W_lin.grad.detach().reshape(-1),
                          dense.bias.grad.detach().reshape(-1)]), None
    if kind == "output":
        return torch.cat([model.output.W_quad.grad.detach().reshape(-1),
                          model.output.W_lin.grad.detach().reshape(-1),
                          model.output.bias.grad.detach().reshape(-1)]), None
    raise ValueError(kind)


def metrics(name, a, b):
    d = (a - b).abs()
    denom = a.abs().clamp_min(1e-12)
    return (name, float(d.max()), float((d / denom).max()), float(d.mean()),
            float(torch.sqrt((d * d).mean())))


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "ref_varseq")
    meta = parse_meta(dirpath)
    vocab = int(meta["vocab_size"])
    emb = int(meta["emb"])
    ff = int(meta["ff"])
    blocks = int(meta["blocks"])
    heads = int(meta["heads"])
    max_seq = int(meta["max_seq"])
    total_rows = int(meta["total_rows"])

    model, _ = build_and_load(vocab, os.path.join(dirpath, "fresh_transformer_weights.bin"),
                              embedding_dimension=emb, feedforward_dimension=ff,
                              transformer_layers=blocks, attention_heads=heads)
    model.train()

    lengths = [int(x) for x in load_ints(os.path.join(dirpath, "batch_seq_lengths.bin"))]
    inputs = load_floats(os.path.join(dirpath, "batch_inputs.bin"))
    correct = load_ints(os.path.join(dirpath, "batch_correct.bin"))
    assert sum(lengths) == total_rows == len(inputs)

    # padded token grid
    tok = np.zeros((len(lengths), max_seq), dtype=np.int64)
    off = 0
    for s, ln in enumerate(lengths):
        tok[s, :ln] = inputs[off:off + ln].astype(np.int64)
        off += ln
    tok_t = torch.from_numpy(tok)
    targets = torch.from_numpy(correct.astype(np.int64))

    out, mask = build_forward_padded(model, tok_t, lengths)
    kinds = layer_kinds(blocks, heads)

    print(f"=== var-length forward ({lengths}), {len(kinds)} layers, "
          f"total_rows={total_rows} ===")
    fails = []
    cpp_loss = float(load_floats(os.path.join(dirpath, "batch_loss.bin"))[0])
    p_target = packed_rows(out["probs"], mask).gather(1, targets.view(-1, 1)).squeeze(1)
    pt_loss = float(torch.mean(-torch.log(p_target + ONE_MINUS_7)))
    print(f"  loss: C++ {cpp_loss:.7f}  PyTorch {pt_loss:.7f}  |diff| {abs(cpp_loss - pt_loss):.3e}")

    max_fwd = 0.0
    for layer_idx, kind in enumerate(kinds):
        b, w, h = block_pos(layer_idx, heads)
        cpp_out = torch.from_numpy(load_floats(
            os.path.join(dirpath, f"ref_layer{layer_idx:02d}_output.bin")).copy())
        pt_out = packed_rows(torch_checkpoint(model, kind, b, h, out), mask).reshape(-1)
        n, ma, mr, me, rms = metrics(f"out{layer_idx}", cpp_out, pt_out)
        max_fwd = max(max_fwd, ma)
        if ma > 1e-5:
            fails.append(f"layer{layer_idx}({kind}) output max_abs={ma:.2e}")

        preact = torch_preact(model, kind, b, h, out)
        if preact is not None:
            cpp_pre = torch.from_numpy(load_floats(
                os.path.join(dirpath, f"ref_layer{layer_idx:02d}_preact.bin")).copy())
            pt_pre = packed_rows(preact, mask).reshape(-1)
            n, ma, mr, me, rms = metrics(f"pre{layer_idx}", cpp_pre, pt_pre)
            max_fwd = max(max_fwd, ma)
            if ma > 1e-5:
                fails.append(f"layer{layer_idx}({kind}) preact max_abs={ma:.2e}")
    print(f"  forward checkpoints: max abs err over all layers/preacts = {max_fwd:.3e}")

    # per-layer input-gradient chain (retain so .grad is populated after backward)
    h_chain = [out["embedding"]]
    for bi2 in range(blocks):
        block = model.blocks[bi2]
        h_chain.append(out[f"block{bi2+1}_norm1"])
        for hi2 in range(heads):
            h_chain.append(out[f"block{bi2+1}_head{hi2+1}_out"])
        h_chain.append(out[f"block{bi2+1}_norm2"])
        h_chain.append(out[f"block{bi2+1}_ff1_relu"])
        h_chain.append(out[f"block{bi2+1}_ff2"])
    h_chain.append(out["final_norm"])
    for h in h_chain:
        h.retain_grad()

    # ---- backward: clean sum-CE over valid rows ----
    loss_sum = F.cross_entropy(
        packed_rows(out["logits"], mask).reshape(-1, vocab), targets, reduction="sum")
    model.zero_grad(set_to_none=True)
    loss_sum.backward()

    print(f"=== var-length backward gradients ===")
    max_grad = 0.0
    for layer_idx, kind in enumerate(kinds):
        b, w, h = block_pos(layer_idx, heads)
        cpp_w = torch.from_numpy(load_floats(
            os.path.join(dirpath, f"ref_layer{layer_idx:02d}_wgrad.bin")).copy())
        pt_w, pt_x = torch_grad_flat(model, kind, b, h)
        n, ma, mr, me, rms = metrics(f"w{layer_idx}", cpp_w, pt_w)
        max_grad = max(max_grad, ma)
        if ma > 1e-4:
            fails.append(f"layer{layer_idx}({kind}) wgrad max_abs={ma:.2e} (rel {mr:.2e})")
        if pt_x is not None:
            cpp_x = torch.from_numpy(load_floats(
                os.path.join(dirpath, f"ref_layer{layer_idx:02d}_extra_wgrad.bin")).copy())
            n, ma, mr, me, rms = metrics(f"x{layer_idx}", cpp_x, pt_x)
            max_grad = max(max_grad, ma)
            if ma > 1e-4:
                fails.append(f"layer{layer_idx}({kind}) extra_wgrad max_abs={ma:.2e}")
    print(f"  weight gradients (all layers): max abs err = {max_grad:.3e}")

    # per-layer input gradients: h_i.grad (valid rows) vs C++ gradin of layer i+1
    max_gin = 0.0
    for i, h in enumerate(h_chain):
        # gradient wrt layer (i+1)'s input = layer i's output
        cpp_gin = torch.from_numpy(load_floats(
            os.path.join(dirpath, f"ref_layer{i+1:02d}_gradin.bin")).copy())
        pt_gin = packed_rows(h.grad, mask).reshape(-1)
        n, ma, mr, me, rms = metrics(f"gin{i+1}", cpp_gin, pt_gin)
        max_gin = max(max_gin, ma)
        if ma > 1e-4:
            fails.append(f"gradin layer{i+1} max_abs={ma:.2e}")
    print(f"  input gradients (layers 1..{len(h_chain)}): max abs err = {max_gin:.3e}")

    # ---- one optimizer step ----
    print(f"=== one optimizer step (step 0, var-length batch) ===")
    B1 = np.float32(0.9); B2 = np.float32(0.999)
    A1 = np.float32(0.1); A2 = np.float32(0.001)
    EPS = np.float32(1e-8)
    LR_EFF = np.float32(0.001) / np.float32(len(lengths))
    WD = np.float32(0.01); QLR = np.float32(0.7); QWD = np.float32(0.7)
    dp = load_floats(os.path.join(dirpath, "decay_powers.bin"))
    den1 = np.float32(1.0) - dp[0]
    den2 = np.float32(1.0) - dp[1]
    m, v = {}, {}
    from optimizer_step import enumerate_params
    for name, p, s, d in enumerate_params(model):
        g = p.grad
        mi = m.get(id(p))
        if mi is None:
            mi = torch.zeros_like(p); m[id(p)] = mi
            vi = torch.zeros_like(p); v[id(p)] = vi
        else:
            vi = v[id(p)]
        mi.mul_(B1).add_(g, alpha=A1)
        vi.mul_(B2).add_(g * g, alpha=A2)
        adam = (mi / den1) / (torch.sqrt(vi / den2) + EPS)
        decay = (WD * d) * p
        p.data.sub_((LR_EFF * s) * (adam + decay))
    parts = [p.detach().reshape(-1) for _, p, _, _ in enumerate_params(model)]
    pt_flat = torch.cat(parts)
    cpp_after = torch.from_numpy(load_floats(
        os.path.join(dirpath, "post_step_weights.bin")).copy())
    n, ma, mr, me, rms = metrics("onestep", cpp_after, pt_flat)
    print(f"  post-step weights: max_abs={ma:.3e} mean_abs={me:.3e} RMS={rms:.3e}")
    if ma > 1e-4:
        fails.append(f"one-step weights max_abs={ma:.2e}")

    print()
    if not fails:
        print(f"== VAR-LENGTH MATCH (loss diff {abs(cpp_loss - pt_loss):.2e}, "
              f"fwd {max_fwd:.2e}, grads {max(max_grad, max_gin):.2e}) ==")
        sys.exit(0)
    print("== VAR-LENGTH MISMATCH ==")
    for f in fails:
        print("  " + f)
    sys.exit(1)


if __name__ == "__main__":
    main()
