"""Diagnostic: interleaved measurement of every phase of one training step.

Not part of the training pipeline and never imported by it. Every phase is run
repeatedly in an interleaved order so machine noise affects them equally, and
median / p90 are reported instead of a single-shot number.

    cd python && python research/profile_first_step.py --iters 20

``trainer._train_batch`` and ``trainer._log_step`` call the real Trainer
methods (no duplicate copies of the step), so the reported "full step" figure
tracks the trainer code it is next to.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from model import Transformer
from research.config import Config
from research.data import iter_batches
from research.tokenizer import build_tokenizer
from research.trainer import Trainer


def stats(samples):
    a = np.asarray(samples) * 1000.0
    return np.median(a), np.percentile(a, 90)


class _NullMetrics:
    """Stands in for MetricsWriter so _log_step does its work but writes nothing."""

    def write(self, **fields):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/TinyStories/tinystories-train.txt")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = args.train if os.path.isabs(args.train) else os.path.join(root, args.train)

    print("== environment ==")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"cpu={os.cpu_count()} threads={torch.get_num_threads()}")

    cfg = Config(max_sequence_length=args.seq_len, batch_size=args.batch_size)
    t0 = time.perf_counter()
    tok = build_tokenizer(train_path, progress=False)
    print(f"tokenizer build: {time.perf_counter() - t0:.1f}s  vocab={tok.vocabulary_size}")
    cfg.vocab_size = tok.vocabulary_size

    t0 = time.perf_counter()
    chunk = next(tok.iter_file_tokens(train_path, progress=False))
    dt = time.perf_counter() - t0
    total_mb = os.path.getsize(train_path) / 1e6
    print(f"tokenize one {len(chunk) / 1e6:.1f}M-char chunk: {dt * 1000:.0f} ms "
          f"-> full {total_mb:.0f} MB pass ~= {dt * total_mb / (len(chunk) / 1e6):.0f}s")

    model = Transformer(cfg.vocab_size, cfg.embedding_dimension,
                        cfg.feedforward_dimension, cfg.transformer_layers,
                        cfg.attention_heads)
    model.init_like_cpp(seed=cfg.seed)
    print(f"parameters={sum(q.numel() for q in model.parameters()):,} "
          f"tensors={sum(1 for _ in model.parameters())}")

    S, B = cfg.max_sequence_length, cfg.batch_size
    trainer = Trainer(model, cfg, tok)
    # steps_per_epoch=1 skips the full-corpus count pass this script does not need
    trainer.prepare(train_path, steps_per_epoch=1)
    opt = trainer.optimizer

    batch_iter = iter_batches(train_path, tok, S, cfg.window_stride, B, progress=False)
    x_np, y_np = next(batch_iter)      # real batch; also pays first chunk tokenization
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    with torch.no_grad():
        acts = trainer._activation_magnitudes(debug=model.forward_debug(x))
    null_metrics = _NullMetrics()

    def forward_only():
        return model.forward(x)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        lg = model.forward(x)
        F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1),
                        reduction="sum").backward()

    def adam_only():
        opt.step()

    def train_batch():
        return trainer._train_batch(x_np, y_np)

    def log_step():
        # logging only: the magnitudes are already captured, so this is the
        # gradient/parameter norm loops plus the metrics row construction
        trainer._log_step(null_metrics, 0, cfg.learning_rate, 5.0, 0.0, acts)

    def data_batch():
        return next(batch_iter)

    phases = {
        "forward": forward_only,
        "fwd+bwd": fwd_bwd,
        "adam.step": adam_only,
        "trainer._train_batch": train_batch,
        "trainer._log_step": log_step,
        "data.next(batch)": data_batch,
    }
    samples = {k: [] for k in phases}
    for fn in phases.values():
        for _ in range(2):
            fn()
    for _ in range(args.iters):
        for k, fn in phases.items():   # interleaved: equal noise exposure
            t0 = time.perf_counter()
            fn()
            samples[k].append(time.perf_counter() - t0)

    print(f"\n== phase timings over {args.iters} interleaved reps (ms) ==")
    for k in phases:
        med, p90 = stats(samples[k])
        print(f"  {k:22s} median {med:8.2f}  p90 {p90:8.2f}")

    med = {k: stats(v)[0] for k, v in samples.items()}
    print("\n== one full trainer step (data + train + per-step logging) ==")
    print(f"  measured median: {med['trainer._train_batch'] + med['data.next(batch)']:.1f} ms")
    print(f"  of which _log_step: {med['trainer._log_step']:.1f} ms")
    print(f"  compute only (no data, no logging): {med['trainer._train_batch']:.1f} ms")
    print(f"  logged activation magnitudes: {acts}")


if __name__ == "__main__":
    main()
