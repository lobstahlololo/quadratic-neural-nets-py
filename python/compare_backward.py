"""Stage-6 backward comparison: PyTorch autograd vs C++ reference gradients.

Uses the same weights/batch as compare_forward.py, computes the sum-CE loss
(C++ gradient semantics: sum over tokens of softmax - onehot), runs backward,
and compares:
  - every layer's input gradient (gradient wrt the layer's input)
  - every dense layer's [W_quad][W_lin][bias] gradients
  - LayerNorm [gamma][beta] gradients
  - attention [Wq][Wk][Wv] gradients
  - embedding gradients

Usage:
    python3 compare_backward.py [ref_data_dir]
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from compare_forward import load_floats, load_ints, parse_meta
from load_transformer import build_and_load

torch.set_num_threads(1)


def ref(dirpath, name):
    return torch.from_numpy(load_floats(os.path.join(dirpath, name)).copy())


def compare(name, cpp_flat, torch_flat, failures):
    cpp = cpp_flat.detach().numpy()
    pt = torch_flat.detach().numpy()
    if cpp.shape != pt.shape:
        raise ValueError(f"{name}: shape mismatch cpp {cpp.shape} vs torch {pt.shape}")
    diff = np.abs(cpp - pt)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    max_abs = float(np.abs(cpp).max())
    flag = "FAIL" if max_err > 1e-3 else "ok"
    print(f"  {flag}  {name:44s} max={max_err:.3e} mean={mean_err:.3e} (cpp max abs={max_abs:.3e})")
    if max_err > 1e-3:
        failures.append((name, max_err, mean_err))


def output_key(layer_index):
    """Key of layer `layer_index`'s output in forward_debug's dict."""
    if layer_index == 0:
        return "embedding"
    if layer_index == 33:
        return "final_norm"
    if layer_index == 34:
        return "logits"
    b = (layer_index - 1) // 8  # block index (0-based)
    r = (layer_index - 1) % 8
    tag = f"block{b+1}"
    if r == 0:
        return f"{tag}_norm1"
    if 1 <= r <= 4:
        return f"{tag}_head{r}_out"
    if r == 5:
        return f"{tag}_norm2"
    if r == 6:
        return f"{tag}_ff1_relu"
    return f"{tag}_ff2"


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

    model, offset = build_and_load(
        vocab,
        os.path.join(dirpath, "fresh_transformer_weights.bin"),
        emb, ff, num_blocks, heads,
    )
    model.train()
    for p in model.parameters():
        p.grad = None

    training_data = load_floats(os.path.join(dirpath, "training_data.bin"))
    correct_indices = load_ints(os.path.join(dirpath, "correct_indices.bin"))
    x = torch.from_numpy(training_data[:total_rows].astype(np.int64)).view(8, max_seq)
    targets = torch.from_numpy(correct_indices[:total_rows].astype(np.int64))

    act = model.forward_debug(x)
    for key, t in act.items():
        if t.requires_grad:
            t.retain_grad()

    # C++ gradient semantics: sum over tokens of (softmax - onehot)
    logits = act["logits"].reshape(-1, vocab)
    loss = F.cross_entropy(logits, targets, reduction="sum")
    loss.backward()

    failures = []
    print("Input-gradient comparisons (gradient wrt each layer's input):")
    for i in range(1, 35):
        cpp = ref(dirpath, f"ref_layer{i:02d}_gradin.bin")
        pt = act[output_key(i - 1)].grad
        compare(f"grad_in_{i} (wrt {output_key(i-1)})", cpp, pt.reshape(-1), failures)

    print("\nWeight-gradient comparisons:")

    def compare_dense(name, dense, wgrad, extra_grad=None, extra_refs=None):
        # wgrad flat layout: [W_quad (out,in)][W_lin (out,in)][bias (out)]
        n = dense.in_features * dense.out_features
        flat = torch.cat(
            [dense.W_quad.grad.reshape(-1), dense.W_lin.grad.reshape(-1), dense.bias.grad.reshape(-1)]
        )
        compare(f"{name}_dense", wgrad, flat, failures)
        if extra_grad is not None and extra_refs is not None:
            ln_w, ln_b = extra_refs
            flat2 = torch.cat([ln_w.grad.reshape(-1), ln_b.grad.reshape(-1)])
            compare(f"{name}_gamma_beta", extra_grad, flat2, failures)

    # output layer (34)
    compare_dense("output", model.output, ref(dirpath, "ref_layer34_wgrad.bin"))

    # final norm (33)
    fn = model.final_norm
    compare_dense("final_norm", fn.dense, ref(dirpath, "ref_layer33_wgrad.bin"),
                  ref(dirpath, "ref_layer33_extra_wgrad.bin"), (fn.ln.weight, fn.ln.bias))

    for b in range(num_blocks):
        block = model.blocks[b]
        tag = f"block{b+1}"
        n1, h0, n2, f1, f2 = 1 + 8 * b, 2 + 8 * b, 6 + 8 * b, 7 + 8 * b, 8 + 8 * b
        compare_dense(f"{tag}_norm1", block.norm1.dense, ref(dirpath, f"ref_layer{n1:02d}_wgrad.bin"),
                      ref(dirpath, f"ref_layer{n1:02d}_extra_wgrad.bin"), (block.norm1.ln.weight, block.norm1.ln.bias))
        for h in range(heads):
            head = block.attention[h]
            cpp = ref(dirpath, f"ref_layer{h0+h:02d}_wgrad.bin")
            flat = torch.cat([head.Wq.grad.reshape(-1), head.Wk.grad.reshape(-1), head.Wv.grad.reshape(-1)])
            compare(f"{tag}_head{h+1}_WqkWv", cpp, flat, failures)
        compare_dense(f"{tag}_norm2", block.norm2.dense, ref(dirpath, f"ref_layer{n2:02d}_wgrad.bin"),
                      ref(dirpath, f"ref_layer{n2:02d}_extra_wgrad.bin"), (block.norm2.ln.weight, block.norm2.ln.bias))
        compare_dense(f"{tag}_ff1", block.ff1, ref(dirpath, f"ref_layer{f1:02d}_wgrad.bin"))
        compare_dense(f"{tag}_ff2", block.ff2, ref(dirpath, f"ref_layer{f2:02d}_wgrad.bin"))

    # embedding (0)
    compare("embedding_weights", ref(dirpath, "ref_layer00_wgrad.bin"),
            model.embedding.weight.grad.reshape(-1), failures)

    print()
    if failures:
        print(f"== FAIL: {len(failures)} gradient checkpoint(s) above threshold ==")
        for name, max_err, _ in failures[:20]:
            print(f"   {name}: max abs err {max_err:.3e}")
        print(f"\nFIRST mismatch: {failures[0][0]} (max abs err {failures[0][1]:.3e})")
        sys.exit(1)
    print("== BACKWARD MATCH (all gradients within 1e-3 max abs error) ==")
    sys.exit(0)


if __name__ == "__main__":
    main()
