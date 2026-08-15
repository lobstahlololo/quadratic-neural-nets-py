"""Stage-5 forward comparison: PyTorch replica vs C++ reference.

Loads the fresh weights dumped by python/export_reference.cpp into the PyTorch
replica, runs the exact same batch-0 input, and compares every intermediate
against the C++ dumps (max abs error + mean abs error per checkpoint).

Usage:
    python3 compare_forward.py [ref_data_dir]
"""

import os
import struct
import sys

import numpy as np
import torch

from load_transformer import build_and_load

torch.set_num_threads(1)


def load_floats(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype="<f4")


def load_ints(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype="<i4")


def parse_meta(dirpath):
    meta = {}
    with open(os.path.join(dirpath, "meta.txt")) as f:
        for line in f:
            if " " in line.strip():
                k, v = line.strip().split()
                meta[k] = v
    return meta


def ref(dirpath, name):
    return torch.from_numpy(load_floats(os.path.join(dirpath, name)).copy())


def ref_seq_batch(dirpath, stem, num_seqs):
    """Assemble a batch tensor from per-sequence dump files (each [128*128] or [128*128])."""
    parts = [
        load_floats(os.path.join(dirpath, f"{stem}_seq{s:02d}.bin")) for s in range(num_seqs)
    ]
    return torch.from_numpy(np.concatenate(parts).copy())


def compare(name, cpp_flat, torch_tensor, total_rows, failures):
    """cpp_flat: np float32 [total_rows*d] or [total_rows*sq]; torch_tensor: [B, S, ...]."""
    cpp = cpp_flat.reshape(total_rows, -1)
    pt = torch_tensor.reshape(total_rows, -1).detach().numpy()
    if cpp.shape != pt.shape:
        raise ValueError(f"{name}: shape mismatch cpp {cpp.shape} vs torch {pt.shape}")
    diff = np.abs(cpp - pt)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    flag = "FAIL" if max_err > 1e-3 else "ok"
    print(f"  {flag}  {name:42s} max={max_err:.3e} mean={mean_err:.3e}")
    if max_err > 1e-3:
        failures.append((name, max_err, mean_err))


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "ref_data"
    )
    meta = parse_meta(dirpath)
    vocab = int(meta["vocab_size"])
    emb = int(meta["emb"])
    ff = int(meta["ff"])
    num_blocks = int(meta["transformer_layers"])
    heads = int(meta["attention_heads"])
    total_rows = int(meta["total_rows"])
    max_seq = int(meta["max_seq"])
    expected_params = int(meta["expected_params"])
    cpp_batch0_loss = float(meta["batch0_loss"])

    print(f"vocab={vocab} emb={emb} ff={ff} blocks={num_blocks} heads={heads}")
    print(f"total_rows={total_rows} expected_params={expected_params}")

    # ---- load fresh C++ weights into the replica ----
    weights_path = os.path.join(dirpath, "fresh_transformer_weights.bin")
    model, offset = build_and_load(vocab, weights_path, emb, ff, num_blocks, heads)
    if offset != expected_params:
        raise SystemExit(f"FATAL: consumed {offset} floats, expected {expected_params}")
    print(f"Weights loaded: {offset:,} floats (matches expected {expected_params:,})\n")
    model.eval()

    # ---- same input batch as the C++ reference ----
    training_data = load_floats(os.path.join(dirpath, "training_data.bin"))
    correct_indices = load_ints(os.path.join(dirpath, "correct_indices.bin"))
    batch_tokens = training_data[:total_rows].astype(np.int64)
    x = torch.from_numpy(batch_tokens).view(8, max_seq)

    with torch.no_grad():
        act = model.forward_debug(x)

    failures = []
    print("Forward comparisons (C++ dump vs PyTorch):")

    # embedding (layer 0)
    compare("embedding", ref(dirpath, "ref_layer00_output.bin"), act["embedding"], total_rows, failures)

    for b in range(num_blocks):
        tag = f"block{b+1}"
        n1, h0, n2, f1, f2 = 1 + 8 * b, 2 + 8 * b, 6 + 8 * b, 7 + 8 * b, 8 + 8 * b
        # norm1: preact = dense output, output = post-LN
        compare(f"{tag}_norm1_dense (pre-LN)",
                ref(dirpath, f"ref_layer{n1:02d}_preact.bin"), act[f"{tag}_norm1_dense"], total_rows, failures)
        compare(f"{tag}_norm1 (post-LN)",
                ref(dirpath, f"ref_layer{n1:02d}_output.bin"), act[f"{tag}_norm1"], total_rows, failures)
        for h in range(heads):
            li = h0 + h
            htag = f"{tag}_head{h+1}"
            compare(f"{htag}_q", ref_seq_batch(dirpath, f"ref_layer{li:02d}_q", 8), act[f"{htag}_q"], total_rows, failures)
            compare(f"{htag}_k", ref_seq_batch(dirpath, f"ref_layer{li:02d}_k", 8), act[f"{htag}_k"], total_rows, failures)
            compare(f"{htag}_v", ref_seq_batch(dirpath, f"ref_layer{li:02d}_v", 8), act[f"{htag}_v"], total_rows, failures)
            compare(f"{htag}_scores (post-softmax)",
                    ref_seq_batch(dirpath, f"ref_layer{li:02d}_scores", 8), act[f"{htag}_scores"], total_rows, failures)
            compare(f"{htag}_out",
                    ref(dirpath, f"ref_layer{li:02d}_output.bin"), act[f"{htag}_out"], total_rows, failures)
        compare(f"{tag}_norm2_dense (pre-LN)",
                ref(dirpath, f"ref_layer{n2:02d}_preact.bin"), act[f"{tag}_norm2_dense"], total_rows, failures)
        compare(f"{tag}_norm2 (post-LN)",
                ref(dirpath, f"ref_layer{n2:02d}_output.bin"), act[f"{tag}_norm2"], total_rows, failures)
        compare(f"{tag}_ff1_dense (pre-ReLU)",
                ref(dirpath, f"ref_layer{f1:02d}_preact.bin"), act[f"{tag}_ff1_dense"], total_rows, failures)
        compare(f"{tag}_ff1_relu (post-ReLU)",
                ref(dirpath, f"ref_layer{f1:02d}_output.bin"), act[f"{tag}_ff1_relu"], total_rows, failures)
        compare(f"{tag}_ff2",
                ref(dirpath, f"ref_layer{f2:02d}_output.bin"), act[f"{tag}_ff2"], total_rows, failures)

    # final norm + output
    compare("final_norm_dense (pre-LN)",
            ref(dirpath, "ref_layer33_preact.bin"), act["final_norm_dense"], total_rows, failures)
    compare("final_norm (post-LN)",
            ref(dirpath, "ref_layer33_output.bin"), act["final_norm"], total_rows, failures)
    compare("logits (pre-softmax)",
            ref(dirpath, "ref_layer34_preact.bin"), act["logits"], total_rows, failures)
    compare("probs (post-softmax)",
            ref(dirpath, "ref_layer34_output.bin"), act["probs"], total_rows, failures)

    # loss: C++ reports mean(-log(p_target + 1e-7))
    probs = act["probs"].reshape(-1, vocab).detach().numpy()
    targets = correct_indices[:total_rows]
    py_loss = float(np.mean(-np.log(probs[np.arange(total_rows), targets] + 1e-7)))
    print(f"\nLoss comparison:")
    print(f"  C++  batch0_loss = {cpp_batch0_loss:.8f}")
    print(f"  PyTorch loss     = {py_loss:.8f}")
    print(f"  diff             = {abs(py_loss - cpp_batch0_loss):.3e}")

    print()
    if failures:
        print(f"== FAIL: {len(failures)} checkpoint(s) above threshold ==")
        for name, max_err, _ in failures:
            print(f"   {name}: max abs err {max_err:.3e}")
        print(f"\nFIRST divergence: {failures[0][0]} (max abs err {failures[0][1]:.3e})")
        sys.exit(1)
    print("== FORWARD MATCH (all checkpoints within 1e-3 max abs error) ==")
    sys.exit(0)


if __name__ == "__main__":
    main()
