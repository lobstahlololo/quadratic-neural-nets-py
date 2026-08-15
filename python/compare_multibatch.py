"""Stage-9 multi-batch comparison: the Bug-D mini-batching path.

Dataset has more than 8 windows (36), so steps_per_epoch > 1 (4) and the
trainScheduler slices training_data into per-step batches, computes the cosine
LR per j (varies across steps), and uses the GLOBAL step number
(epoch*steps_per_epoch + j) for Adam bias correction.

The C++ side (robust_harness mode=train) dumped per-step loss, per-step LR and
per-step float32 bias-correction powers plus weight snapshots. This script
replays the identical loop in PyTorch and compares:
  - per-step reported loss
  - flat parameter vector at snapshot steps (1,2,5,10,N)
  - final moment buffers

Usage: python3 compare_multibatch.py <ref_dir>
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from compare_forward import load_floats, load_ints, parse_meta
from load_transformer import build_and_load
from optimizer_step import enumerate_params

torch.set_num_threads(1)

ONE_MINUS_7 = np.float32(1e-7)
B1 = np.float32(0.9); B2 = np.float32(0.999)
A1 = np.float32(0.1); A2 = np.float32(0.001)
EPS = np.float32(1e-8)
WD = np.float32(0.01)


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "ref_multibatch")
    meta = parse_meta(dirpath)
    vocab = int(meta["vocab_size"])
    emb = int(meta["emb"])
    ff = int(meta["ff"])
    blocks = int(meta["blocks"])
    heads = int(meta["heads"])
    max_seq = int(meta["max_seq"])
    batch_size = int(meta["batch_size"])
    epochs = int(meta["epochs"])
    sps = int(meta["steps_per_epoch"])
    num_steps = epochs * sps

    model, _ = build_and_load(vocab, os.path.join(dirpath, "fresh_transformer_weights.bin"),
                              embedding_dimension=emb, feedforward_dimension=ff,
                              transformer_layers=blocks, attention_heads=heads)
    model.train()

    training = load_floats(os.path.join(dirpath, "training_data.bin"))
    correct = load_ints(os.path.join(dirpath, "correct_indices.bin"))
    targets_flat = load_floats(os.path.join(dirpath, "targets.bin"))
    rows = batch_size * max_seq

    lrs = []
    with open(os.path.join(dirpath, "multistep_lrs.txt")) as f:
        for line in f:
            lrs.append(float(line.split()[1]))
    assert len(lrs) == num_steps, f"lrs file {len(lrs)} vs num_steps {num_steps}"
    dp = load_floats(os.path.join(dirpath, "multistep_decay_powers.bin")).reshape(num_steps, 2)

    cpp_losses = []
    with open(os.path.join(dirpath, "multistep_losses.txt")) as f:
        for line in f:
            cpp_losses.append(float(line.split()[1]))

    m, v = {}, {}
    pt_losses = []
    snapshots = {}
    for gs in range(num_steps):
        j = gs % sps
        x = torch.from_numpy(training[j * rows:(j + 1) * rows].astype(np.int64)).view(batch_size, max_seq)
        tgt = torch.from_numpy(correct[j * rows:(j + 1) * rows].astype(np.int64))
        act = model.forward_debug(x)
        logits = act["logits"].reshape(-1, vocab)
        probs = act["probs"].reshape(-1, vocab)

        p_target = probs.gather(1, tgt.view(-1, 1)).squeeze(1)
        pt_losses.append(float(torch.mean(-torch.log(p_target + ONE_MINUS_7))))

        loss_sum = F.cross_entropy(logits, tgt, reduction="sum")
        model.zero_grad(set_to_none=True)
        loss_sum.backward()

        lr = np.float32(lrs[gs])
        lr_eff = lr / np.float32(batch_size)
        den1 = np.float32(1.0) - dp[gs, 0]
        den2 = np.float32(1.0) - dp[gs, 1]
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
            p.data.sub_((lr_eff * s) * (adam + decay))

        after = gs + 1
        if after in (1, 2, 5, 10, 20, 50) or after == num_steps:
            snapshots[after] = torch.cat(
                [p.detach().reshape(-1) for _, p, _, _ in enumerate_params(model)]).numpy()

    # ---- per-step loss ----
    print(f"=== multi-batch training (windows={meta['num_windows']}, sps={sps}, "
          f"epochs={epochs}, steps={num_steps}) ===")
    max_loss_diff = 0.0
    print(f"  {'step':>4s} {'j':>2s} {'lr':>11s} {'C++ loss':>11s} {'PyTorch':>11s} {'|diff|':>10s}")
    for gs in range(num_steps):
        d = abs(cpp_losses[gs] - pt_losses[gs])
        max_loss_diff = max(max_loss_diff, d)
        if gs < 6 or gs >= num_steps - 2:
            print(f"  {gs+1:4d} {gs % sps:2d} {lrs[gs]:11.7f} {cpp_losses[gs]:11.7f} "
                  f"{pt_losses[gs]:11.7f} {d:10.3e}")
    print(f"  max |loss diff| over {num_steps} steps: {max_loss_diff:.3e}")

    # ---- parameter vectors ----
    print("\n=== flat parameter vector (C++ vs PyTorch, snapshot steps) ===")
    maxabs_series = []
    for after in sorted(snapshots):
        cpp = load_floats(os.path.join(dirpath, f"post_step_{after:02d}_weights.bin"))
        pt = snapshots[after]
        d = np.abs(cpp - pt)
        ma, me, rms = float(d.max()), float(d.mean()), float(np.sqrt(np.mean(d * d)))
        maxabs_series.append(ma)
        print(f"  step {after:3d}: max_abs={ma:.3e} mean_abs={me:.3e} RMS={rms:.3e}")
    if len(maxabs_series) >= 2:
        ratios = [maxabs_series[i] / max(maxabs_series[i - 1], 1e-30)
                  for i in range(1, len(maxabs_series))]
        geom = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-30, None)))))
        print(f"  growth ratios: " + " ".join(f"{r:.2f}" for r in ratios))
        print(f"  geometric mean growth ratio per snapshot: {geom:.3f} "
              f"({'STABLE' if geom < 3.0 else 'DIVERGING'})")
    else:
        geom = 1.0

    # ---- final moments ----
    m_flat = torch.cat([m[id(p)].reshape(-1) for _, p, _, _ in enumerate_params(model)]).numpy()
    v_flat = torch.cat([v[id(p)].reshape(-1) for _, p, _, _ in enumerate_params(model)]).numpy()
    cpp_m1 = load_floats(os.path.join(dirpath, "final_first_moments.bin"))
    cpp_m2 = load_floats(os.path.join(dirpath, "final_second_moments.bin"))
    print("\n=== final-step moment buffers ===")
    for lbl, cpp, pt in (("first ", cpp_m1, m_flat), ("second", cpp_m2, v_flat)):
        d = np.abs(cpp - pt)
        print(f"  {lbl} max_abs={float(d.max()):.3e} mean_abs={float(d.mean()):.3e}")

    stable = geom < 3.0
    if max_loss_diff < 1e-3 and max(maxabs_series) < 1e-2 and stable:
        print(f"\n== MULTI-BATCH TRAINING MATCH (max loss diff {max_loss_diff:.2e}, "
              f"max param diff {max(maxabs_series):.2e}) ==")
        sys.exit(0)
    print(f"\n== MULTI-BATCH MISMATCH (max loss diff {max_loss_diff:.2e}, "
          f"max param diff {max(maxabs_series):.2e}, stable={stable}) ==")
    sys.exit(1)


if __name__ == "__main__":
    main()
