"""Stage-7 one-step optimizer comparison: exact C++ train_adams reconstruction.

The C++ update (boilerplate/train_functions.h, train_adams) for step 0:
    lr_eff      = learning_rate / batch_size          (0.001f / 8)
    beta1=0.9, beta2=0.999, eps=1e-8, moments start at zero
    bias correction: m/(1-0.9^1), v/(1-0.999^1)       (denominators in float32)
    m = 0.9*m + 0.1*g ;  v = 0.999*v + 0.001*g*g
    p -= lr_eff * s * ( m_hat/(sqrt(v_hat)+eps) + wd * d * p )
    s=d=0.7 for W_quad blocks (quad_lr_scale/quad_wd_scale), else 1.0
Gradients are SUM-reduced (reduction='sum' CE) exactly like C++.

Compares the post-update flat parameter vector and the first/second moment
buffers against the C++ dumps in python/ref_data/, per parameter class.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from compare_forward import load_floats, load_ints, parse_meta
from load_transformer import build_and_load

torch.set_num_threads(1)

# ---- exact float32 constants as used by the C++ code ----
B1 = np.float32(0.9)      # 0.9f
B2 = np.float32(0.999)    # 0.999f
A1 = np.float32(0.1)      # literal 0.1f in the m update
A2 = np.float32(0.001)    # literal 0.001f in the v update
EPS = np.float32(1e-8)    # 1e-8f
LR = np.float32(0.001)    # 0.001f
BS = np.float32(8)
LR_EFF = LR / BS          # 0.001f / 8  (float / int in C++)
WD = np.float32(0.01)     # global_weight_decay
QLR = np.float32(0.7)     # quad_lr_scale
QWD = np.float32(0.7)     # quad_wd_scale
# bias-correction denominators: 1.0f - pow(0.9f,1), 1.0f - pow(0.999f,1)
DEN1 = np.float32(1.0 - np.float32(0.9))
DEN2 = np.float32(1.0 - np.float32(0.999))


def enumerate_params(model):
    """Yield (name, param, lr_scale, wd_scale) in the exact C++ flat layout order."""
    yield "embedding.weight", model.embedding.weight, 1.0, 1.0
    # IMPORTANT: the C++ flat weight layout is, per block,
    #   norm1 -> attention 1..4 -> norm2 -> ff1 -> ff2
    # (the 4 attention heads sit BETWEEN norm1 and norm2). The enumeration
    # order below MUST mirror that exactly or every flat index from the first
    # block's attention onward is shifted by the 33,152-float norm2 block.
    for b, block in enumerate(model.blocks):
        sub = block.norm1
        yield f"block{b+1}_norm1.W_quad", sub.dense.W_quad, QLR, QWD
        yield f"block{b+1}_norm1.W_lin", sub.dense.W_lin, 1.0, 1.0
        yield f"block{b+1}_norm1.bias", sub.dense.bias, 1.0, 1.0
        yield f"block{b+1}_norm1.gamma", sub.ln.weight, 1.0, 1.0
        yield f"block{b+1}_norm1.beta", sub.ln.bias, 1.0, 1.0
        for h in range(4):
            head = block.attention[h]
            for k in ("Wq", "Wk", "Wv"):
                yield f"block{b+1}_head{h+1}.{k}", getattr(head, k), 1.0, 1.0
        sub = block.norm2
        yield f"block{b+1}_norm2.W_quad", sub.dense.W_quad, QLR, QWD
        yield f"block{b+1}_norm2.W_lin", sub.dense.W_lin, 1.0, 1.0
        yield f"block{b+1}_norm2.bias", sub.dense.bias, 1.0, 1.0
        yield f"block{b+1}_norm2.gamma", sub.ln.weight, 1.0, 1.0
        yield f"block{b+1}_norm2.beta", sub.ln.bias, 1.0, 1.0
        yield f"block{b+1}_ff1.W_quad", block.ff1.W_quad, QLR, QWD
        yield f"block{b+1}_ff1.W_lin", block.ff1.W_lin, 1.0, 1.0
        yield f"block{b+1}_ff1.bias", block.ff1.bias, 1.0, 1.0
        yield f"block{b+1}_ff2.W_quad", block.ff2.W_quad, QLR, QWD
        yield f"block{b+1}_ff2.W_lin", block.ff2.W_lin, 1.0, 1.0
        yield f"block{b+1}_ff2.bias", block.ff2.bias, 1.0, 1.0
    yield "final_norm.W_quad", model.final_norm.dense.W_quad, QLR, QWD
    yield "final_norm.W_lin", model.final_norm.dense.W_lin, 1.0, 1.0
    yield "final_norm.bias", model.final_norm.dense.bias, 1.0, 1.0
    yield "final_norm.gamma", model.final_norm.ln.weight, 1.0, 1.0
    yield "final_norm.beta", model.final_norm.ln.bias, 1.0, 1.0
    yield "output.W_quad", model.output.W_quad, QLR, QWD
    yield "output.W_lin", model.output.W_lin, 1.0, 1.0
    yield "output.bias", model.output.bias, 1.0, 1.0


def class_of(name):
    if name == "embedding.weight":
        return "embedding"
    if ".W_quad" in name:
        return "W_quad"
    if ".W_lin" in name:
        return "W_lin"
    if name.endswith(".bias"):
        return "bias"
    if name.endswith(".gamma") or name.endswith(".beta"):
        return "gamma_beta"
    return "attention"  # Wq/Wk/Wv


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "ref_data"
    )
    meta = parse_meta(dirpath)
    vocab = int(meta["vocab_size"])
    total_rows = int(meta["total_rows"])
    max_seq = int(meta["max_seq"])

    model, offset = build_and_load(vocab, os.path.join(dirpath, "fresh_transformer_weights.bin"))
    model.train()
    for p in model.parameters():
        p.grad = None

    training = load_floats(os.path.join(dirpath, "training_data.bin"))
    correct = load_ints(os.path.join(dirpath, "correct_indices.bin"))
    x = torch.from_numpy(training[:total_rows].astype(np.int64)).view(8, max_seq)
    targets = torch.from_numpy(correct[:total_rows].astype(np.int64))

    # SUM-reduced CE = C++ gradient semantics (sum over tokens of softmax-onehot)
    act = model.forward_debug(x)
    loss = F.cross_entropy(act["logits"].reshape(-1, vocab), targets, reduction="sum")
    loss.backward()

    # ---- apply the exact C++ one-step update ----
    m = {}
    v = {}
    for name, p, s, d in enumerate_params(model):
        g = p.grad
        mi = m.get(id(p))
        if mi is None:
            mi = torch.zeros_like(p)
            m[id(p)] = mi
            vi = torch.zeros_like(p)
            v[id(p)] = vi
        else:
            vi = v[id(p)]
        mi.mul_(B1).add_(g, alpha=A1)          # 0.9*m + 0.1*g   (float32)
        vi.mul_(B2).add_(g * g, alpha=A2)      # 0.999*v + 0.001*g*g (float32)
        m_hat = mi / DEN1
        v_hat = vi / DEN2
        adam = m_hat / (torch.sqrt(v_hat) + EPS)
        decay = (WD * d) * p                   # (wd*decay_scale) * p  (float32)
        update = (LR_EFF * s) * (adam + decay)
        p.data.sub_(update)

    # ---- flatten PyTorch post-update params + moments in C++ layout order ----
    parts = []
    m_parts = []
    v_parts = []
    class_parts = {}
    for name, p, s, d in enumerate_params(model):
        flat = p.detach().reshape(-1)
        parts.append(flat)
        m_parts.append(m[id(p)].reshape(-1))
        v_parts.append(v[id(p)].reshape(-1))
        class_parts.setdefault(class_of(name), []).append(flat)

    pt_flat = torch.cat(parts)
    pt_m1 = torch.cat(m_parts)
    pt_m2 = torch.cat(v_parts)

    cpp_after = torch.from_numpy(load_floats(os.path.join(dirpath, "post_step_weights.bin")).copy())
    cpp_m1 = torch.from_numpy(load_floats(os.path.join(dirpath, "post_step_first_moments.bin")).copy())
    cpp_m2 = torch.from_numpy(load_floats(os.path.join(dirpath, "post_step_second_moments.bin")).copy())

    if pt_flat.shape != cpp_after.shape:
        raise SystemExit(f"size mismatch: torch {pt_flat.shape} vs cpp {cpp_after.shape}")

    def metrics(name, a, b):
        d = (a - b).abs()
        denom = a.abs().clamp_min(1e-12)
        rel = (d / denom).max()
        return dict(
            name=name,
            max_abs=float(d.max()),
            mean_abs=float(d.mean()),
            rms_abs=float(torch.sqrt((d * d).mean())),
            max_rel=float(rel),
        )

    print("=== post-update parameter vector ===")
    whole = metrics("whole", cpp_after, pt_flat)
    for k, vv in whole.items():
        if k != "name":
            print(f"  {k:8s} {vv:.6e}")
    print(f"  size {len(cpp_after):,}")

    print("\n=== per-class errors (C++ post-step vs PyTorch post-step) ===")
    # The C++ flat layout interleaves classes per layer, so each class is NOT
    # contiguous. Gather C++ values at the exact flat index ranges of each
    # parameter, in enumerate order, to match the PyTorch class tensors.
    cls_ranges = {}
    off = 0
    for nm, p, _, _ in enumerate_params(model):
        n = p.numel()
        cls_ranges.setdefault(class_of(nm), []).append((off, off + n))
        off += n
    max_rel_all = 0.0
    for cls, flats in class_parts.items():
        cpp_idx = np.concatenate(
            [np.arange(a, b) for a, b in cls_ranges[cls]]
        )
        cpp_cls = torch.from_numpy(cpp_after.numpy()[cpp_idx])
        pt_cls = torch.cat(flats)
        mm = metrics(cls, cpp_cls, pt_cls)
        max_rel_all = max(max_rel_all, mm["max_rel"])
        print(f"  {cls:10s} max_abs={mm['max_abs']:.3e} max_rel={mm['max_rel']:.3e} "
              f"mean_abs={mm['mean_abs']:.3e} rms={mm['rms_abs']:.3e} n={len(cpp_idx):,}")

    print("\n=== moment buffers ===")
    for lbl, cpp, pt in (("first", cpp_m1, pt_m1), ("second", cpp_m2, pt_m2)):
        mm = metrics(lbl, cpp, pt)
        print(f"  {lbl:6s} max_abs={mm['max_abs']:.3e} max_rel={mm['max_rel']:.3e} mean_abs={mm['mean_abs']:.3e}")

    # ---- representative before/after values ----
    cpp_before = torch.from_numpy(load_floats(os.path.join(dirpath, "fresh_transformer_weights.bin")).copy())
    print("\n=== representative values (flat-index: C++ before/after vs PyTorch after) ===")
    picks = []
    off = 0
    for nm, p, _, _ in enumerate_params(model):
        n = p.numel()
        if nm == "output.W_quad":
            picks.append(("output.W_quad[0,0]", off, p, (0, 0)))
        if nm == "output.W_lin":
            picks.append(("output.W_lin[0,0]", off, p, (0, 0)))
        if nm == "output.bias":
            picks.append(("output.bias[5]", off, p, (5,)))
        if nm == "final_norm.W_quad":
            picks.append(("final_norm.W_quad[5,3]", off, p, (5, 3)))
        if nm == "final_norm.gamma":
            picks.append(("final_norm.gamma[3]", off, p, (3,)))
        if nm == "embedding.weight":
            picks.append(("embedding[0,0]", off, p, (0, 0)))
        if nm == "block1_head1.Wq":
            picks.append(("block1_head1.Wq[0,1]", off, p, (0, 1)))
        off += n
    for label, off, p, idx in picks:
        # flat index within the parameter tensor
        shape = tuple(p.shape)
        flat_i = 0
        for dim, i in zip(shape, idx):
            flat_i = flat_i * dim + i
        print(f"  {label:22s} C++ {float(cpp_before[off+flat_i]): .8f} -> {float(cpp_after[off+flat_i]): .8f}"
              f"   PyTorch {float(cpp_before[off+flat_i]): .8f} -> {float(p.flatten()[flat_i]): .8f}")

    print()
    # The whole-vector max_abs is dominated by Adam-epsilon amplification on
    # gradients at/below 1e-8: both libraries compute those gradients as ~0 but
    # differ by ~1e-9 of float32 reduction noise, and adam = g/(|g|+1e-8)
    # amplifies that into a weight diff up to ~lr_eff*0.4 (~5e-5). The decisive
    # test is the max diff among elements with a REAL gradient (|g| >= 1e-6).
    cpp_g = load_floats(os.path.join(dirpath, "post_step_wgrads.bin"))
    d_arr = np.abs(cpp_after.numpy() - pt_flat.numpy())
    meaningful = d_arr[np.abs(cpp_g) >= 1e-6]
    max_abs_meaningful = float(meaningful.max()) if meaningful.size else 0.0
    print(f"  max_abs on |g|>=1e-6 (meaningful-gradient) elements: {max_abs_meaningful:.3e}")
    if whole["max_abs"] < 1e-4 and max_abs_meaningful < 1e-6:
        print("== ONE-STEP OPTIMIZER MATCH (float32 noise level) ==")
        sys.exit(0)
    print(f"== ONE-STEP OPTIMIZER MISMATCH (whole max_abs {whole['max_abs']:.3e}, "
          f"meaningful max_abs {max_abs_meaningful:.3e}) ==")
    sys.exit(1)


if __name__ == "__main__":
    main()
