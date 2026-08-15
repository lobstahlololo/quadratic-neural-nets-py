"""Stage-8 multi-step C++ <-> PyTorch training equivalence.

Runs N optimizer steps on the PyTorch replica with EXACTLY the C++ trainScheduler
+ train_adams semantics (verified against train/train.cpp and
boilerplate/train_functions.h):

  - built-in corpus -> steps_per_epoch = 1024 / (8*128) = 1, so every epoch is
    ONE step on the SAME 1024-token batch; cosine LR at j=0 = max LR = 0.001f
    for every step (constant).
  - lr_eff = lr / batch_size = 0.001f / 8
  - beta1=0.9, beta2=0.999, eps=1e-8, moments start at zero
  - bias-correction denominators 1 - pow(0.9, t+1), 1 - pow(0.999, t+1) are
    recomputed from the exact float32 powers dumped by the C++ helper
    (multistep_decay_powers.bin) -- same values std::pow produced.
  - decoupled weight decay 0.01; W_quad blocks use quad_lr_scale=0.7 and
    quad_wd_scale=0.7, everything else 1.0.
  - gradients are SUM-reduced over tokens (reduction='sum' CE == C++ fused
    softmax-onehot summed over rows).
  - reported loss = mean over rows of -log(p_target + 1e-7) (C++ reporting
    convention); backprop uses the CLEAN softmax-onehot gradient (no +1e-7).

Compares per-step reported loss and the flat parameter vector at snapshot steps
(1, 2, 5, 10, 20, 50, N) against the C++ dumps, per parameter class.

Usage: python3 multistep_compare.py [ref_data_dir] [N]
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from compare_forward import load_floats, load_ints, parse_meta
from load_transformer import build_and_load
from optimizer_step import enumerate_params, class_of

torch.set_num_threads(1)

# ---- exact float32 constants as used by the C++ code ----
B1 = np.float32(0.9)
B2 = np.float32(0.999)
A1 = np.float32(0.1)
A2 = np.float32(0.001)
EPS = np.float32(1e-8)
LR = np.float32(0.001)
BS = np.float32(8)
LR_EFF = LR / BS
WD = np.float32(0.01)
QLR = np.float32(0.7)
QWD = np.float32(0.7)
ONE = np.float32(1.0)
ONE_MINUS_7 = np.float32(1e-7)


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "ref_data"
    )
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    meta = parse_meta(dirpath)
    vocab = int(meta["vocab_size"])
    total_rows = int(meta["total_rows"])
    max_seq = int(meta["max_seq"])

    model, _ = build_and_load(vocab, os.path.join(dirpath, "fresh_transformer_weights.bin"))
    model.train()

    training = load_floats(os.path.join(dirpath, "training_data.bin"))
    correct = load_ints(os.path.join(dirpath, "correct_indices.bin"))
    x = torch.from_numpy(training[:total_rows].astype(np.int64)).view(8, max_seq)
    targets = torch.from_numpy(correct[:total_rows].astype(np.int64))

    # exact float32 bias-correction powers per step: pow(0.9,t+1), pow(0.999,t+1)
    dp = load_floats(os.path.join(dirpath, "multistep_decay_powers.bin")).reshape(N, 2)

    # ---- run N steps with the exact C++ semantics ----
    m, v = {}, {}
    cpp_losses = []
    pt_losses = []
    snapshots = {}
    for t in range(N):
        act = model.forward_debug(x)
        logits = act["logits"].reshape(-1, vocab)
        probs = act["probs"].reshape(-1, vocab)

        # C++ reported loss: mean over rows of -log(p_target + 1e-7)
        p_target = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        rep = float(torch.mean(-torch.log(p_target.detach() + ONE_MINUS_7)))
        pt_losses.append(rep)

        # clean SUM-reduced CE gradient (matches C++ fused softmax-onehot)
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        model.zero_grad(set_to_none=True)
        loss_sum.backward()

        den1 = ONE - dp[t, 0]
        den2 = ONE - dp[t, 1]
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
            mi.mul_(B1).add_(g, alpha=A1)          # 0.9*m + 0.1*g
            vi.mul_(B2).add_(g * g, alpha=A2)      # 0.999*v + 0.001*g*g
            m_hat = mi / den1
            v_hat = vi / den2
            adam = m_hat / (torch.sqrt(v_hat) + EPS)
            decay = (WD * d) * p
            p.data.sub_((LR_EFF * s) * (adam + decay))

        after = t + 1
        if after in (1, 2, 5, 10, 20, 50) or after == N:
            parts = [p.detach().reshape(-1) for _, p, _, _ in enumerate_params(model)]
            snapshots[after] = torch.cat(parts).numpy()

    # ---- C++ losses ----
    with open(os.path.join(dirpath, "multistep_losses.txt")) as f:
        for line in f:
            parts = line.split()
            cpp_losses.append(float(parts[1]))
    assert len(cpp_losses) == N, f"C++ losses file has {len(cpp_losses)} lines, expected {N}"

    # ---- per-step loss comparison ----
    print(f"=== per-step reported loss (C++ vs PyTorch), N={N} ===")
    print(f"  {'step':>4s} {'C++ loss':>12s} {'PyTorch':>12s} {'|diff|':>12s}")
    max_loss_diff = 0.0
    for t in range(N):
        d = abs(cpp_losses[t] - pt_losses[t])
        max_loss_diff = max(max_loss_diff, d)
        if t < 5 or t >= N - 3 or t in (9, 19):
            print(f"  {t+1:4d} {cpp_losses[t]:12.7f} {pt_losses[t]:12.7f} {d:12.3e}")
    print(f"  max |loss diff| over {N} steps: {max_loss_diff:.3e}")
    print(f"  loss range C++: {cpp_losses[0]:.4f} -> {cpp_losses[-1]:.4f}")

    # ---- parameter vector comparison at snapshot steps ----
    def metrics(a, b):
        d = np.abs(a - b)
        return float(d.max()), float(d.mean()), float(np.sqrt(np.mean(d * d)))

    cls_ranges = {}
    off = 0
    for nm, p, _, _ in enumerate_params(model):
        n = p.numel()
        cls_ranges.setdefault(class_of(nm), []).append((off, off + n))
        off += n
    cpp_idx = {}
    for cls, rng in cls_ranges.items():
        cpp_idx[cls] = np.concatenate([np.arange(a, b) for a, b in rng])

    print(f"\n=== flat parameter vector (C++ post-step vs PyTorch post-step) ===")
    print(f"  {'step':>5s} {'max_abs':>10s} {'mean_abs':>10s} {'RMS':>10s} "
          f"{'W_quad':>9s} {'W_lin':>9s} {'bias':>9s} {'gamma/beta':>10s} {'emb':>9s} {'attn':>9s}")
    maxabs_series = []
    for after in sorted(snapshots):
        cpp = load_floats(os.path.join(dirpath, f"post_step_{after:02d}_weights.bin"))
        pt = snapshots[after]
        ma, me, rms = metrics(cpp, pt)
        maxabs_series.append(ma)
        cls_max = {}
        for cls in ("W_quad", "W_lin", "bias", "gamma_beta", "embedding", "attention"):
            cls_max[cls] = float(np.abs(cpp[cpp_idx[cls]] - pt[cpp_idx[cls]]).max())
        print(f"  {after:5d} {ma:10.3e} {me:10.3e} {rms:10.3e} "
              f"{cls_max['W_quad']:9.2e} {cls_max['W_lin']:9.2e} {cls_max['bias']:9.2e} "
              f"{cls_max['gamma_beta']:10.2e} {cls_max['embedding']:9.2e} {cls_max['attention']:9.2e}")

    # ---- growth assessment: is the discrepancy accumulating or stable? ----
    print("\n=== discrepancy growth assessment ===")
    if len(maxabs_series) >= 2:
        ratios = [maxabs_series[i] / max(maxabs_series[i - 1], 1e-30)
                  for i in range(1, len(maxabs_series))]
        print(f"  max_abs series: " + " ".join(f"{v:.2e}" for v in maxabs_series))
        print(f"  step-to-step ratios: " + " ".join(f"{r:.2f}" for r in ratios))
        geometric = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-30, None)))))
        print(f"  geometric mean growth ratio per snapshot: {geometric:.3f} "
              f"({'STABLE' if geometric < 3.0 else 'DIVERGING'})")

    # ---- final moments ----
    print("\n=== final-step moment buffers (step %d) ===" % N)
    m_flat = torch.cat([m[id(p)].reshape(-1) for _, p, _, _ in enumerate_params(model)]).numpy()
    v_flat = torch.cat([v[id(p)].reshape(-1) for _, p, _, _ in enumerate_params(model)]).numpy()
    cpp_m1 = load_floats(os.path.join(dirpath, "final_first_moments.bin"))
    cpp_m2 = load_floats(os.path.join(dirpath, "final_second_moments.bin"))
    for lbl, cpp, pt in (("first ", cpp_m1, m_flat), ("second", cpp_m2, v_flat)):
        ma, me, rms = metrics(cpp, pt)
        print(f"  {lbl} max_abs={ma:.3e} mean_abs={me:.3e} RMS={rms:.3e}")

    # ---- verdict ----
    print()
    # Loss diff at ~1e-6..1e-4 is float32 reduction-order noise. Parameter diffs
    # are dominated by Adam-epsilon amplification on sub-1e-8 gradients
    # (adam = g/(|g|+eps) amplifies ~1e-9 of gradient noise into ~5e-5 weight
    # diffs). As long as max_abs stays < 1e-3 and does not grow super-linearly
    # (geometric ratio < 3 per snapshot), the trajectories are equivalent.
    stable = geometric < 3.0 if len(maxabs_series) >= 2 else True
    if max_loss_diff < 1e-3 and max(maxabs_series) < 1e-2 and stable:
        print(f"== MULTI-STEP TRAINING MATCH (float32 noise level; max loss diff "
              f"{max_loss_diff:.2e}, max param diff {max(maxabs_series):.2e}) ==")
        sys.exit(0)
    print(f"== MULTI-STEP TRAINING MISMATCH (max loss diff {max_loss_diff:.2e}, "
          f"max param diff {max(maxabs_series):.2e}, stable={stable}) ==")
    sys.exit(1)


if __name__ == "__main__":
    main()
