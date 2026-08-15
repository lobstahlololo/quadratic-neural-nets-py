# Quadratic Transformer — Research Setup (Stage 10)

This package is the research harness for the **C++-validated quadratic
transformer** (`quadratic-neuralnets`). Every mathematical detail below is
either the exact C++ reference behavior or a deliberate, documented research
choice. The C++ reference, the PyTorch replica in `python/model.py`, and the
audit scripts in `python/` are **frozen**; the equivalence was validated in
Stages 1–9 (forward, backward, one-step Adam, 100-step training, variable
length, toy finite differences, second seed).

## 1. Model

### Quadratic layer
Every dense layer computes

```
y = x @ W_lin.T + (x*x) @ W_quad.T + bias
```

`QuadraticLinear(in, out)` stores `W_quad (out,in)`, `W_lin (out,in)`,
`bias (out)`. The C++ flat layout per dense layer is exactly
`[W_quad][W_lin][bias]`.

### QuadNorm
`QuadraticLinear(dim, dim) -> LayerNorm(dim, eps=1e-5)` (C++ `norm` block: a
Quadratic layer whose output is normalized). LayerNorm is per-token over
features with biased variance and eps=1e-5.

### Attention
`SingleHeadAttention(dim)`: **single-head** causal attention.

```
q = x @ Wq ; k = x @ Wk ; v = x @ Wv        (no bias; x @ W, not x @ W.T)
scores = (q @ k.T) / sqrt(dim)
causal mask = -1e30 above the diagonal
attn = softmax(scores) ; out = attn @ v
```

There are **no residual connections anywhere**. The C++ runs
`attention_heads` sequential single-head attentions per block (this is NOT
multi-head attention — heads do not share the input in parallel).

### Architecture (default config)
```
Embedding(vocab, emb)
-> transformer_layers blocks:
     QuadNorm
     -> attention_heads x SingleHeadAttention (sequential)
     -> QuadNorm
     -> QuadraticLinear(emb -> ff) -> ReLU
     -> QuadraticLinear(ff -> emb)
-> final QuadNorm
-> QuadraticLinear(emb -> vocab)   (softmax applied by the C++ hook / loss)
```
Defaults: emb=128, ff=512, 4 blocks, 4 heads, max_seq=128.

## 2. Loss and reduction

- **Backpropagation** uses **sum-reduced** cross entropy
  (`F.cross_entropy(..., reduction="sum")`), reproducing the C++ per-token
  softmax–onehot gradient **sum**. This is required for C++ equivalence;
  `reduction="mean"` would silently halve/scale all gradients.
- **Reported loss** is the C++ value: `mean(-log(p_target + 1e-7))` over the
  batch rows (the `+1e-7` is part of the C++ `CrossEntropyLossForSoftmax` and
  is reproduced exactly).

## 3. Optimizer

`torch.optim.AdamW` with **explicit parameter groups** (see `optimizer.py`,
`params.py`). The grouping is configuration-agnostic (derives block/head
counts from the model), not hardcoded to 4 heads.

```
lr_eff  = learning_rate / batch_size          (C++ scaled_learning_rate)
beta1=0.9, beta2=0.999, eps=1e-8
normal params:  lr = lr_eff,            weight_decay = 0.01
W_quad params:  lr = quad_scale*lr_eff, weight_decay = quad_scale*0.01
```

This is algebraically identical to the C++ `train_adams` update

```
p -= lr_eff * s * (m_hat/(sqrt(v_hat)+eps) + wd * d * p)   s=d=quad_scale on W_quad
```

The 0.7 scale on W_quad exists because the quadratic path is |x| times more
sensitive to weight changes than the linear path (`d y/d W_quad = x^2` vs
`d y/d W_lin = x`), so a slightly reduced LR/decay equalises the effective
function-space step size. The one-step update was verified against the C++
post-step weight dump (max abs diff 7.9e-6 whole vector, 2.2e-8 on elements
with a real gradient — float32 noise only).

With `--no-quadratic`, all `W_quad` are zeroed and **excluded** from the
optimizer (they stay exactly zero).

## 4. Scheduler

Exact C++ cosine, **per-epoch**:

```
lr(j) = min_lr + (lr - min_lr) * (1 + cos(pi * j / steps_per_epoch)) / 2
```

- `j` is the step inside the current epoch, `j in [0, steps_per_epoch)`; it
  resets at the start of every epoch.
- `global_step = epoch * steps_per_epoch + j` is separate and drives the Adam
  bias correction (torch's AdamW tracks it internally as its step counter;
  it equals the C++ global step because we step exactly once per training
  step and never reset the optimizer).

## 5. Data (streaming)

`data.py` reproduces `documentation/example_transformer.cpp` windowing:
windows of length S slide with stride S/2; each window yields
`inputs[p:p+S]`, `targets[p+1:p+S+1]` (next-token pairs) while
`p + S < T` (the C++ loop guard). `training_data` is the concatenation of all
windows; a batch is `batch_size` consecutive windows, and
`steps_per_epoch = num_windows // batch_size` (integer division, trailing
windows dropped) — exactly C++ `training_data.size() / total_input_rows`.

The 1.9 GB train corpus is **never loaded into memory**: the tokenizer and
the window/batch iterators stream the file in 8 MiB chunks with a sliding
token buffer (bounded by ~stride+S+chunk_size). No shuffle by default.

### Variable-length batches
`model.forward_packed(tokens, lengths)` runs the validated padded-grid
forward (causal mask + padding mask, softmax over exactly the valid key
columns) and returns `(out, mask)`; it is bit-identical to the Stage-9
`compare_varseq.build_forward_padded` used for C++ equivalence.

## 6. Seeding, checkpointing, metrics

- `seed_everything(seed)` seeds Python, NumPy and PyTorch RNGs. The model is
  initialized with `init_like_cpp(seed)` (C++ xavier init, embedding-first,
  local seeded generator — the seed fully determines the init).
- Checkpoints (`checkpoint.py`) contain model, optimizer, scheduler state,
  `global_step`, `epoch`, and Python/NumPy/PyTorch (incl. CUDA) RNG states.
  Resume restores all of them, so the next step is deterministic.
- Metrics (`metrics.py`) are written as JSONL with one row per logged step:
  step, epoch, train_loss, val_loss, learning_rate, W_quad/W_lin/other
  gradient norms, W_quad/W_lin parameter norms, and activation magnitudes.

## 7. CLI

```
python3 research/research_train.py --train data/TinyStories/tinystories-train.txt \
    --valid data/TinyStories/tinystories-valid.txt --epochs 100 --output-dir runs/exp1
```

See `--help` for all options (seed, device, dims, blocks, heads, batch-size,
seq-len, lr, min-lr, weight-decay, quad-scale, quadratic on/off, output-dir,
resume, max-steps-per-epoch for controlled runs).

## 8. Validation status

- Forward: ~110 checkpoints match C++ to ≤8e-7; loss diff ~1e-5.
- Backward: gradients match C++ (W_quad fix verified by finite differences).
- One-step Adam: matches C++ post-step weights to float32 noise.
- 100-step training: loss diff ≤2.6e-5, parameter trajectories track C++.
- Variable-length, toy-FD (9.7e-10), multi-batch scheduler, second seed: pass.
- Known float32 noise: gradients at/below 1e-8 get Adam-epsilon amplified
  (weight diff up to ~5e-5 for those elements); the decisive metric is the
  max diff on elements with |g| ≥ 1e-6 (~1e-8).
- Stage-9 limitations: `compare_backward.py`'s absolute 1e-3 threshold
  marginally flags `output_dense` under seed 7 (1.8e-5 relative — threshold
  artifact, not semantic); BPE is O(n²) so the audit suite used the reference
  corpus, not a 128-token-window TinyStories subset.

## 9. Self-checks

`python3 research/self_check.py` verifies: A same-seed init, B different-seed
init, C deterministic one step, D checkpoint resume determinism, E scheduler
C++ values, F `quadratic=False` zeroing/exclusion, G tiny end-to-end run,
H streaming pipeline on the real 1.9 GB corpus without loading it whole.
