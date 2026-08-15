"""Research training CLI for the quadratic transformer.

Example (tiny smoke run on the real valid file as a stand-in corpus):

    python -m research.research_train --train data/TinyStories/tinystories-valid.txt \
        --valid data/TinyStories/tinystories-valid.txt --seq-len 64 --epochs 2 \
        --output-dir runs/smoke

Full config defaults match the validated C++ reference (see config.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# make `python3 research/research_train.py` and `python3 -m research.research_train`
# both work from the python/ directory: put python/ on sys.path and import the
# research package absolutely so its relative sibling imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.config import Config  # noqa: E402
from research.tokenizer import build_tokenizer  # noqa: E402
from research.data import steps_per_epoch  # noqa: E402
from research.trainer import Trainer, seed_everything  # noqa: E402
from model import Transformer  # noqa: E402


def build_model(config, tokenizer):
    from research.params import zero_w_quad  # noqa: E402
    model = Transformer(
        vocabulary_size=config.vocab_size,
        embedding_dimension=config.embedding_dimension,
        feedforward_dimension=config.feedforward_dimension,
        transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
    )
    model.init_like_cpp(seed=config.seed)
    if not config.quadratic:
        zero_w_quad(model)
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the C++-validated quadratic transformer on TinyStories.")
    p.add_argument("--train", default=None, help="training corpus path")
    p.add_argument("--valid", default=None, help="validation corpus path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--min-lr", type=float, default=0.00001)
    p.add_argument("--quadratic", dest="quadratic", action="store_true", default=True)
    p.add_argument("--no-quadratic", dest="quadratic", action="store_false")
    p.add_argument("--quad-scale", type=float, default=0.7)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--emb", type=int, default=128, dest="embedding_dimension")
    p.add_argument("--ff", type=int, default=512, dest="feedforward_dimension")
    p.add_argument("--blocks", type=int, default=4, dest="transformer_layers")
    p.add_argument("--heads", type=int, default=4, dest="attention_heads")
    p.add_argument("--max-steps-per-epoch", type=int, default=0,
                   help="cap optimizer steps per epoch (0 = full epoch)")
    p.add_argument("--max-valid-batches", type=int, default=0,
                   help="cap validation batches (0 = all)")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(
        seed=args.seed,
        device=args.device,
        embedding_dimension=args.embedding_dimension,
        feedforward_dimension=args.feedforward_dimension,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        max_sequence_length=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        quad_scale=args.quad_scale,
        quadratic=args.quadratic,
        epochs=args.epochs,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
    )
    seed_everything(cfg.seed)

    train_path = args.train or cfg.train_path
    valid_path = args.valid or cfg.valid_path
    if not os.path.exists(train_path):
        raise SystemExit(f"train corpus not found: {train_path}")

    print(f"[tokenizer] building vocabulary from {train_path}")
    tokenizer = build_tokenizer(train_path)
    cfg.vocab_size = tokenizer.vocabulary_size
    print(f"[tokenizer] vocabulary size = {cfg.vocab_size}")

    model = build_model(cfg, tokenizer)
    print(f"[model] parameters = {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, cfg, tokenizer, device=cfg.device)
    sps, nw, total = steps_per_epoch(
        train_path, tokenizer, cfg.max_sequence_length, cfg.window_stride,
        cfg.batch_size, progress=True)
    print(f"[data] tokens={total:,} windows={nw:,} steps_per_epoch={sps}")

    metrics_path, final_ckpt = trainer.run(
        train_path,
        valid_path=valid_path if os.path.exists(valid_path) else None,
        epochs=cfg.epochs,
        log_interval=cfg.log_interval,
        checkpoint_dir=cfg.output_dir,
        epochs_per_checkpoint=cfg.epochs_per_checkpoint,
        max_steps_per_epoch=args.max_steps_per_epoch,
        max_valid_batches=args.max_valid_batches,
        resume_path=args.resume,
    )
    print(f"[done] metrics={metrics_path} checkpoint={final_ckpt}")


if __name__ == "__main__":
    main()
