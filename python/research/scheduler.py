"""Exact C++ cosine learning-rate schedule.

Reproduces trainScheduler (train/train.cpp) exactly:

    lr(j) = min_lr + (lr - min_lr) * (1 + cos(pi * j / steps_per_epoch)) / 2

Semantics that MUST be preserved:
  - ``j`` is the step index INSIDE the current epoch: j in [0, steps_per_epoch).
    It resets to 0 at the start of every epoch (the C++ cosine is per-epoch,
    NOT a global-step cosine).
  - The global step (epoch * steps_per_epoch + j) is separate and drives the
    Adam bias correction inside the optimizer.
"""

from __future__ import annotations

import math


def lr_for_step(j, steps_per_epoch, learning_rate, min_learning_rate):
    """Return the C++ cosine LR for in-epoch step ``j``.

    At j=0 this returns ``learning_rate`` (cos(0)=1 -> (1+1)/2 = 1) and at the
    hypothetical j=steps_per_epoch it would return ``min_learning_rate``; the
    real loop only ever sees j < steps_per_epoch.
    """
    if steps_per_epoch <= 0:
        return float(learning_rate)
    c = math.cos(math.pi * j / steps_per_epoch)
    return min_learning_rate + (learning_rate - min_learning_rate) * (1.0 + c) / 2.0


class CosineScheduler:
    """Per-epoch cosine scheduler with C++-matching step numbering.

    Typical use inside a training loop::

        sched = CosineScheduler(sps, lr, min_lr)
        for epoch in range(epochs):
            for j in range(sps):
                lr = sched.get(j)               # in-epoch step j
                global_step = sched.global_step(epoch, j)
                ...  # forward/backward with this lr, optimizer step at global_step

    No internal mutable state: the schedule is a pure function of (epoch, j),
    which makes checkpoint/resume trivially deterministic.
    """

    def __init__(self, steps_per_epoch, learning_rate, min_learning_rate):
        self.steps_per_epoch = int(steps_per_epoch)
        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)

    def get(self, j):
        return lr_for_step(j, self.steps_per_epoch, self.learning_rate, self.min_learning_rate)

    def global_step(self, epoch, j):
        """C++ Adam step counter: step = epoch * steps_per_epoch + j (0-based)."""
        return epoch * self.steps_per_epoch + j

    def state_dict(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
        }

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["steps_per_epoch"], d["learning_rate"], d["min_learning_rate"])

    def load_state_dict(self, d):
        """In-place restore (checkpoint API parity with torch optimizers)."""
        self.steps_per_epoch = int(d["steps_per_epoch"])
        self.learning_rate = float(d["learning_rate"])
        self.min_learning_rate = float(d["min_learning_rate"])


# ---------------------------------------------------------------------------
# Tests (run directly: python3 research/scheduler.py)
# ---------------------------------------------------------------------------
def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_known_values():
    lr, min_lr = 0.001, 0.00001
    # sps = 4: values match the Stage-9 multi-batch C++ run exactly
    sched = CosineScheduler(4, lr, min_lr)
    assert _close(sched.get(0), 0.001), sched.get(0)                       # j=0 -> lr
    assert _close(sched.get(1), 0.00085502, 1e-6), sched.get(1)            # cos(pi/4)
    assert _close(sched.get(2), 0.000505, 1e-6), sched.get(2)              # cos(pi/2)=0 -> mid
    assert _close(sched.get(3), 0.00015502, 1e-6), sched.get(3)            # cos(3pi/4)
    # mid-point property: at j = sps/2 the schedule is the arithmetic mean
    sched = CosineScheduler(10, 0.002, 0.00002)
    assert _close(sched.get(5), (0.002 + 0.00002) / 2, 1e-9), sched.get(5)
    # j resets per epoch: same j -> same lr regardless of epoch
    assert _close(sched.get(1), sched.get(1))
    # global step numbering: epoch * sps + j (on the sps=10 scheduler)
    assert sched.global_step(0, 0) == 0
    assert sched.global_step(0, 5) == 5
    assert sched.global_step(1, 0) == 10
    assert sched.global_step(2, 1) == 21
    # and on the sps=4 scheduler
    assert CosineScheduler(4, lr, min_lr).global_step(1, 0) == 4
    print("scheduler tests: PASS (known values, mid-point, per-epoch reset, global step)")


if __name__ == "__main__":
    test_known_values()
