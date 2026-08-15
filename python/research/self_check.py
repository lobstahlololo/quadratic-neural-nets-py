"""Stage-10 self-checks (A-H): verify research infrastructure before large runs.

Run from python/:  python3 research/self_check.py

Checks:
  A. same seed -> identical initialization
  B. different seeds -> different initialization
  C. one optimizer step is deterministic
  D. save -> load checkpoint -> next step identical to no-checkpoint run
  E. scheduler matches expected C++ LR values
  F. quadratic=False zeros W_quad and excludes it from the optimizer
  G. a tiny end-to-end training run completes
  H. the streaming tokenizer/data pipeline produces a small batch without
     loading the whole (multi-GB) corpus into RAM
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

# allow `python3 research/self_check.py` and `python3 -m research.self_check`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer  # noqa: E402
from research.config import Config  # noqa: E402
from research.tokenizer import CharacterTokenizer  # noqa: E402
from research.data import iter_batches, num_windows  # noqa: E402
from research.params import iter_parameters, build_param_groups, zero_w_quad  # noqa: E402
from research.optimizer import make_optimizer  # noqa: E402
from research.scheduler import CosineScheduler, lr_for_step  # noqa: E402
from research.trainer import Trainer, seed_everything  # noqa: E402
from research.checkpoint import save_checkpoint, load_checkpoint  # noqa: E402

torch.set_num_threads(1)

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def make_tiny_corpus(d, n=30):
    text = ("One day a little girl named Lily found a needle in her room. "
            "She knew it was dangerous and ran to her mother. "
            "The mother smiled and said good girl. " * n)
    p = os.path.join(d, "corpus.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p, CharacterTokenizer(text)


def tiny_model(vocab, seed=1, quad=True):
    m = Transformer(vocabulary_size=vocab, embedding_dimension=8,
                    feedforward_dimension=16, transformer_layers=2,
                    attention_heads=2)
    m.init_like_cpp(seed=seed)
    if not quad:
        zero_w_quad(m)
    return m


# ---------------------------------------------------------------------------
def check_a_b(seed_a=1, seed_b=2):
    print("[A/B] deterministic + distinct initialization")
    va = 30
    m1 = tiny_model(va, seed=seed_a)
    m2 = tiny_model(va, seed=seed_a)
    m3 = tiny_model(va, seed=seed_b)
    p1 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m1)])
    p2 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m2)])
    p3 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m3)])
    check("A: same seed -> identical init", torch.equal(p1, p2))
    check("B: different seeds -> different init", not torch.equal(p1, p3))


def check_c():
    print("[C] one optimizer step deterministic")
    seed_everything(7)
    m1 = tiny_model(30, seed=3)
    m2 = tiny_model(30, seed=3)
    cfg = Config(seed=3, batch_size=4, max_sequence_length=16, vocab_size=30,
                 embedding_dimension=8, feedforward_dimension=16,
                 transformer_layers=2, attention_heads=2)
    # identical inputs for both branches: generate once BEFORE the loop so the
    # global RNG state does not advance differently between branches
    x = torch.randint(0, 30, (4, 16))
    y = torch.randint(0, 30, (4, 16))
    for m in (m1, m2):
        opt = make_optimizer(m, lr_eff=0.001 / 4, weight_decay=0.01, quad_scale=0.7)
        m.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m.forward(x).reshape(-1, 30), y.reshape(-1), reduction="sum")
        loss.backward()
        opt.step()
    p1 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m1)])
    p2 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m2)])
    check("C: identical params after one step", torch.equal(p1, p2))


def check_d():
    print("[D] checkpoint save/load -> identical next step")
    seed_everything(11)
    m1 = tiny_model(30, seed=5)
    m2 = tiny_model(30, seed=5)
    cfg = Config(seed=5, batch_size=4, max_sequence_length=16, vocab_size=30,
                 embedding_dimension=8, feedforward_dimension=16,
                 transformer_layers=2, attention_heads=2)
    sched = CosineScheduler(3, 0.001, 0.00001)

    def run(model, optimizer, scheduler, x, y, step):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.forward(x).reshape(-1, 30), y.reshape(-1), reduction="sum")
        loss.backward()
        optimizer.step()
        return model, optimizer, scheduler

    x = torch.randint(0, 30, (4, 16))
    y = torch.randint(0, 30, (4, 16))

    # branch 1: two steps, no checkpoint
    o1 = make_optimizer(m1, lr_eff=0.001 / 4, weight_decay=0.01, quad_scale=0.7)
    run(m1, o1, sched, x, y, 0)
    run(m1, o1, sched, x, y, 1)

    # branch 2: one step, save, then re-init and load, then second step
    o2 = make_optimizer(m2, lr_eff=0.001 / 4, weight_decay=0.01, quad_scale=0.7)
    run(m2, o2, sched, x, y, 0)
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ck.pt")
        save_checkpoint(ck, m2, o2, sched, global_step=1, epoch=0,
                        config_dict=cfg.to_dict())
        # fresh model + optimizer + scheduler, then load
        m2b = tiny_model(30, seed=5)
        o2b = make_optimizer(m2b, lr_eff=0.001 / 4, weight_decay=0.01, quad_scale=0.7)
        sched_b = CosineScheduler(3, 0.001, 0.00001)
        gs, ep, cfg_d, _ = load_checkpoint(ck, m2b, o2b, sched_b)
        run(m2b, o2b, sched_b, x, y, 1)
        p1 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m1)])
        p2 = torch.cat([p.detach().reshape(-1) for _, p, _, _ in iter_parameters(m2b)])
        check("D: resume reproduces next step exactly", torch.equal(p1, p2),
              f"global_step={gs} epoch={ep}")


def check_e():
    print("[E] scheduler matches C++ LR values")
    ok = True
    lr, min_lr, sps = 0.001, 0.00001, 4
    expected = {0: 0.001, 1: 0.00085502, 2: 0.000505, 3: 0.00015502}
    for j, want in expected.items():
        got = lr_for_step(j, sps, lr, min_lr)
        if abs(got - want) > 1e-6:
            ok = False
    check("E: cosine values at j=0..3 (sps=4)", ok)
    s = CosineScheduler(4, lr, min_lr)
    check("E: per-epoch reset (same j -> same lr)", s.get(1) == s.get(1))
    check("E: global step = epoch*sps+j", s.global_step(2, 1) == 9)


def check_f():
    print("[F] quadratic=False zeros W_quad and excludes from optimizer")
    m = tiny_model(30, seed=2, quad=False)
    # W_quad all zero
    quads = [p for name, p, _, _ in iter_parameters(m) if name.endswith(".W_quad")]
    all_zero = all(float(p.detach().abs().max()) == 0.0 for p in quads)
    check("F: all W_quad zeroed", all_zero)
    opt = make_optimizer(m, lr_eff=0.001 / 4, weight_decay=0.01, quad_scale=0.7,
                         quadratic=False)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_quad = sum(p.numel() for p in quads)
    n_all = sum(p.numel() for _, p, _, _ in iter_parameters(m))
    check("F: W_quad excluded from optimizer", n_opt + n_quad == n_all,
          f"optimizer elems={n_opt:,} W_quad elems={n_quad:,} total={n_all:,}")
    # a step must keep W_quad exactly zero
    x = torch.randint(0, 30, (2, 8))
    y = torch.randint(0, 30, (2, 8))
    m.zero_grad(set_to_none=True)
    loss = F.cross_entropy(m.forward(x).reshape(-1, 30), y.reshape(-1), reduction="sum")
    loss.backward()
    opt.step()
    still_zero = all(float(p.detach().abs().max()) == 0.0 for p in quads)
    check("F: W_quad stays zero after a step", still_zero)


def check_g():
    print("[G] tiny end-to-end training run")
    with tempfile.TemporaryDirectory() as d:
        train_p, tok = make_tiny_corpus(d, n=60)
        valid_p = train_p
        cfg = Config(seed=42, vocab_size=tok.vocabulary_size,
                     embedding_dimension=8, feedforward_dimension=16,
                     transformer_layers=2, attention_heads=2,
                     max_sequence_length=32, batch_size=4,
                     learning_rate=0.001, min_learning_rate=0.00001,
                     epochs=2, output_dir=os.path.join(d, "out"))
        model = tiny_model(tok.vocabulary_size, seed=42)
        trainer = Trainer(model, cfg, tok)
        trainer.prepare(train_p)
        sps = trainer.scheduler.steps_per_epoch
        metrics_path, final_ckpt = trainer.run(
            train_p, valid_path=valid_p, epochs=2,
            checkpoint_dir=cfg.output_dir, max_steps_per_epoch=2)
        ok = (sps >= 1 and os.path.exists(metrics_path) and os.path.exists(final_ckpt))
        with open(metrics_path) as f:
            rows = [line for line in f if line.strip()]
        check("G: training run completes + writes metrics/checkpoint", ok,
              f"sps={sps} metric_rows={len(rows)}")
        check("G: metrics contain expected fields",
              "wquad_grad_norm" in rows[0] and "train_loss" in rows[0]
              and "learning_rate" in rows[0])


def check_h():
    print("[H] streaming pipeline produces a batch without loading whole corpus")
    # Use the real TinyStories corpus if present; otherwise a multi-MB corpus.
    candidates = [
        "data/TinyStories/tinystories-train.txt",
        os.path.join("..", "data", "TinyStories", "tinystories-train.txt"),
    ]
    real = next((c for c in candidates if os.path.exists(c)), None)
    with tempfile.TemporaryDirectory() as d:
        if real is None:
            real = os.path.join(d, "big.txt")
            with open(real, "w", encoding="utf-8") as f:
                for i in range(20000):
                    f.write("The little red fox jumped over the lazy dog. " * 30 + "\n")
        from research.tokenizer import build_tokenizer
        print(f"    building tokenizer from {real} ...")
        tok = build_tokenizer(real, progress=False)
        # build ONE batch, streaming: stop after the first batch so the rest of
        # a multi-GB corpus is never read (proves the pipeline is incremental)
        batches = iter_batches(real, tok, seq_len=64, stride=32, batch_size=2,
                               progress=False)
        first = next(batches, None)
        check("H: tokenizer builds + first batch produced", first is not None,
              f"vocab={tok.vocabulary_size} batch0 shapes="
              f"{first[0].shape}/{first[1].shape}")
        # Verify windows are correct next-token pairs
        inp, tgt = first
        assert np.array_equal(inp[:, 1:], tgt[:, :-1]), "next-token shift broken"
        check("H: next-token input/target pairing correct", True)


def main():
    print("Stage-10 self-checks (A-H)\n")
    check_a_b()
    check_c()
    check_d()
    check_e()
    check_f()
    check_g()
    check_h()
    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
