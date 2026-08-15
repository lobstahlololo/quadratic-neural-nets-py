"""Standalone research training loop (C++ trainScheduler-compatible).

Implements the exact validated training semantics:

  - Deterministic seeding (Python / NumPy / PyTorch RNG).
  - Data: fixed-length windows of seq_len tokens, batch_size consecutive
    windows per batch, no shuffle, re-streamed per epoch (memory-bounded).
  - Gradients are SUM-reduced over tokens (F.cross_entropy reduction='sum'),
    exactly like the C++ softmax-onehot per-token gradient sum. The lr is
    divided by batch_size to form lr_eff (C++: scaled_learning_rate =
    learning_rate / batch_size).
  - Reported loss is the C++ value: mean over rows of -log(p_target + 1e-7).
  - Scheduler: C++ per-epoch cosine lr(j) = min + (max-min)(1+cos(pi*j/sps))/2,
    j in [0, sps), reset each epoch; global step epoch*sps+j drives the Adam
    bias correction (torch.optim.AdamW tracks this internally as its step
    counter, which equals the C++ global step because we step exactly once
    per training step and never reset the optimizer).
  - Metrics per step: train/val loss, lr, gradient/parameter/update norms
    split into W_quad / W_lin / other, and activation magnitudes.
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from .checkpoint import save_checkpoint, load_checkpoint
from .metrics import MetricsWriter
from .optimizer import make_optimizer
from .params import iter_parameters, class_of
from .scheduler import CosineScheduler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ONE_MINUS_7 = 1e-7


def reported_loss(probs, targets):
    """C++ reported loss: mean over rows of -log(p_target + 1e-7).

    probs may be (B, S, V) or (N, V); targets (B, S) or (N,) respectively.
    """
    probs_flat = probs.reshape(-1, probs.shape[-1])
    targets_flat = targets.reshape(-1)
    p_target = probs_flat.gather(1, targets_flat.view(-1, 1)).squeeze(1)
    return float(torch.mean(-torch.log(p_target + ONE_MINUS_7)).detach())


def _norm_of(params):
    total = 0.0
    for p in params:
        if p is not None:
            total += float(p.detach().pow(2).sum())
    return math.sqrt(total)


class Trainer:
    def __init__(self, model, config, tokenizer, device=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device or config.device)
        self.model.to(self.device)
        seed_everything(config.seed)

        self.lr_eff = config.learning_rate / config.batch_size
        self.scheduler = CosineScheduler(
            1, config.learning_rate, config.min_learning_rate)  # sps fixed below
        self.optimizer = None
        self.global_step = 0
        self.epoch = 0

    # -- data -----------------------------------------------------------------
    def _sps(self, train_path):
        return data_mod.steps_per_epoch(
            train_path, self.tokenizer, self.config.max_sequence_length,
            self.config.window_stride, self.config.batch_size,
            progress=False)[0]

    def prepare(self, train_path):
        """Compute steps_per_epoch and (re)create optimizer + scheduler."""
        sps = self._sps(train_path)
        self.scheduler = CosineScheduler(
            sps, self.config.learning_rate, self.config.min_learning_rate)
        self.optimizer = make_optimizer(
            self.model, self.lr_eff, weight_decay=self.config.weight_decay,
            quad_scale=self.config.quad_scale, beta1=self.config.beta1,
            beta2=self.config.beta2, epsilon=self.config.epsilon,
            quadratic=self.config.quadratic)
        return sps

    def _train_batch(self, inputs, targets, max_grad_norm=None):
        x = torch.from_numpy(inputs).to(self.device)
        y = torch.from_numpy(targets).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model.forward(x)
        # gradient loss: SUM-reduced CE (C++ per-token gradient sum)
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum")
        loss_sum.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        probs = torch.softmax(logits, dim=-1)
        return reported_loss(probs, y), loss_sum.item()

    def _validate(self, valid_path, max_batches=0):
        """Mean reported loss over validation batches (no gradient)."""
        self.model.eval()
        losses = []
        count = 0
        with torch.no_grad():
            for inputs, targets in data_mod.iter_batches(
                    valid_path, self.tokenizer, self.config.max_sequence_length,
                    self.config.window_stride, self.config.batch_size,
                    progress=False):
                x = torch.from_numpy(inputs).to(self.device)
                y = torch.from_numpy(targets).to(self.device)
                logits = self.model.forward(x)
                probs = torch.softmax(logits, dim=-1)
                losses.append(reported_loss(probs, y))
                count += 1
                if max_batches and count >= max_batches:
                    break
        self.model.train()
        return float(np.mean(losses)) if losses else float("nan")

    def _activation_magnitudes(self, inputs):
        """Mean |activation| of key intermediates for the current batch."""
        x = torch.from_numpy(inputs).to(self.device)
        with torch.no_grad():
            out = self.model.forward_debug(x)
        mags = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                mags[k] = float(v.abs().mean())
        return mags

    def run(self, train_path, valid_path=None, epochs=None, log_interval=1,
            checkpoint_dir=None, epochs_per_checkpoint=1, max_steps_per_epoch=0,
            max_valid_batches=0, resume_path=None):
        """Full training loop; returns (metrics_path, final_checkpoint_path)."""
        epochs = epochs if epochs is not None else self.config.epochs
        log_interval = log_interval or self.config.log_interval
        epochs_per_checkpoint = epochs_per_checkpoint or self.config.epochs_per_checkpoint

        os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None
        metrics = MetricsWriter(os.path.join(checkpoint_dir, "metrics.jsonl"))

        # compute steps_per_epoch and create the optimizer (needed before
        # any checkpoint can be loaded into it)
        if self.optimizer is None:
            self.prepare(train_path)
        if resume_path:
            global_step, epoch, cfg_dict, extra = load_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler,
                device=self.device)
            self.global_step = global_step
            self.epoch = epoch
            print(f"[resume] step={self.global_step} epoch={self.epoch}")

        self.model.train()
        sps = self.scheduler.steps_per_epoch
        if max_steps_per_epoch:
            sps = min(sps, max_steps_per_epoch)

        start_epoch = self.epoch
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            epoch_losses = []
            # re-stream the corpus once per epoch (memory-bounded): the batch
            # generator yields consecutive windows in C++ order; step j takes
            # batch j of the epoch (mirrors trainScheduler slicing).
            batch_iter = data_mod.iter_batches(
                train_path, self.tokenizer, self.config.max_sequence_length,
                self.config.window_stride, self.config.batch_size,
                progress=False)
            for j in range(sps):
                try:
                    inputs, targets = next(batch_iter)
                except StopIteration:
                    break
                lr_now = self.scheduler.get(j)
                # C++: scaled_learning_rate = learning_rate / batch_size applied
                # to the scheduled LR, then the W_quad group gets the 0.7 scale
                # via its group lr (quad_scale * lr_eff). Keep group ratios
                # intact by scaling the shared lr_eff component.
                base = lr_now / self.config.batch_size
                q = self.config.quad_scale
                self.optimizer.param_groups[0]["lr"] = base
                if len(self.optimizer.param_groups) > 1:
                    self.optimizer.param_groups[1]["lr"] = q * base
                tr_loss, sum_loss = self._train_batch(inputs, targets)
                self.global_step += 1
                epoch_losses.append(tr_loss)
                if (self.global_step % log_interval == 0):
                    self._log_step(metrics, j, lr_now, tr_loss, sum_loss, inputs)
            # -- end of epoch --
            self.epoch = epoch + 1
            avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_loss = None
            if valid_path:
                val_loss = self._validate(valid_path, max_batches=max_valid_batches)
            print(f"[epoch {self.epoch}/{epochs}] train_loss={avg:.6f} "
                  f"val_loss={val_loss if val_loss is None else round(val_loss, 6)} "
                  f"({time.time()-epoch_start:.1f}s)")
            metrics.write(step=self.global_step, epoch=self.epoch,
                          train_loss=avg, val_loss=val_loss,
                          learning_rate=self.scheduler.get(0))
            if checkpoint_dir and (self.epoch % epochs_per_checkpoint == 0):
                ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch{self.epoch}.pt")
                self.save(ckpt)
                print(f"[saved] {ckpt}")
        metrics.close()
        final = None
        if checkpoint_dir:
            final = os.path.join(checkpoint_dir, "checkpoint_final.pt")
            self.save(final)
        return os.path.join(checkpoint_dir, "metrics.jsonl") if checkpoint_dir else None, final

    def _log_step(self, metrics, j, lr_now, tr_loss, sum_loss, inputs):
        # gradient norms (from .grad) classified W_quad / W_lin / other
        g_quad, g_lin, g_other = [], [], []
        for name, p, _, _ in iter_parameters(self.model):
            if p.grad is None:
                continue
            cls = class_of(name)
            (g_quad if cls == "W_quad" else g_lin if cls == "W_lin" else g_other).append(p.grad)
        p_quad, p_lin = [], []
        for name, p, _, _ in iter_parameters(self.model):
            cls = class_of(name)
            if cls == "W_quad":
                p_quad.append(p.detach())
            elif cls == "W_lin":
                p_lin.append(p.detach())
        activations = self._activation_magnitudes(inputs)
        metrics.write(
            step=self.global_step, epoch=self.epoch, train_loss=tr_loss,
            val_loss=None, learning_rate=lr_now,
            wquad_grad_norm=_norm_of(g_quad), wlin_grad_norm=_norm_of(g_lin),
            other_grad_norm=_norm_of(g_other),
            wquad_param_norm=_norm_of(p_quad), wlin_param_norm=_norm_of(p_lin),
            activations={k: v for k, v in activations.items()
                         if k in ("embedding", "final_norm", "logits")},
            sum_ce_loss=sum_loss)

    def save(self, path):
        save_checkpoint(path, self.model, self.optimizer, self.scheduler,
                        self.global_step, self.epoch, self.config.to_dict())
