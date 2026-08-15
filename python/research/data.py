"""Streaming TinyStories data pipeline (C++ trainScheduler-compatible).

Reproduces documentation/example_transformer.cpp windowing exactly:

  - The corpus is tokenized to a flat token stream (streaming, never materialized).
  - Windows: for position p in 0, stride, 2*stride, ... while p + S < T:
        inputs[p..p+S)   targets[p+1..p+S+1]      (next-token pairs)
    where T = total token count, S = seq_len, stride = S/2 by default.
    The condition p + S < T is the C++ `position + maximum_sequence_length
    < token_ids.size()` loop guard: a window is emitted only once token p+S
    exists (its last target).
  - training_data in C++ is the concatenation of all windows; a batch is
    batch_size*S consecutive tokens of that stream = batch_size consecutive
    windows. steps_per_epoch = num_windows // batch_size (integer division,
    trailing windows ignored), exactly like
    `steps_per_epoch = training_data.size() / total_input_rows`.

Memory: the 1.9 GB train corpus is never loaded whole. Windows are emitted
from a sliding buffer that holds at most ~stride+S+chunk_size token ids, and
batches are assembled batch_size windows at a time.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .tokenizer import build_tokenizer, DEFAULT_CHUNK_SIZE


def num_windows(token_count, seq_len, stride):
    """Number of windows the C++ loop would emit for a token stream of this length."""
    if token_count <= seq_len:
        return 0
    # p in 0, stride, 2*stride ... while p + seq_len < token_count
    return (token_count - seq_len - 1) // stride + 1


def _stream(path, tokenizer, chunk_size, progress_cb):
    """Iterate token-id chunks, invoking progress_cb(read_bytes, total) per chunk."""
    return tokenizer.iter_file_tokens(path, chunk_size=chunk_size, progress=False,
                                      progress_cb=progress_cb)


def count_tokens(path, tokenizer, chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
    """Stream the file through the tokenizer and return the total token count."""
    total_bytes = os.path.getsize(path)
    last_pct = -1
    total = 0
    label = "counting {}".format(os.path.basename(path))

    def cb(read, total_bytes=total_bytes):
        nonlocal last_pct
        if progress and total_bytes > 0:
            pct = int(read * 100 // total_bytes)
            if pct > last_pct:
                sys.stderr.write("\r{}: {}% ({:.0f} / {:.0f} MB)".format(
                    label, pct, read / 1e6, total_bytes / 1e6))
                sys.stderr.flush()
                last_pct = pct

    for chunk in _stream(path, tokenizer, chunk_size, cb):
        total += len(chunk)
    if progress and total_bytes > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()
    return total


def steps_per_epoch(path, tokenizer, seq_len, stride, batch_size,
                    chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
    """C++ steps_per_epoch = len(training_data) / (batch_size * seq_len).

    Counts tokens with one streaming pass (never loads the corpus), then
    divides num_windows by batch_size. Returns (steps_per_epoch, num_windows,
    total_tokens).
    """
    total = count_tokens(path, tokenizer, chunk_size=chunk_size, progress=progress)
    nw = num_windows(total, seq_len, stride)
    return nw // batch_size, nw, total


def iter_windows(path, tokenizer, seq_len, stride, chunk_size=DEFAULT_CHUNK_SIZE,
                 progress=True):
    """Yield (inputs, targets) numpy int64 arrays of length seq_len, streaming.

    Matches the C++ window loop; bounded memory via a sliding buffer.
    """
    total_bytes = os.path.getsize(path)
    last_pct = -1
    label = "windowizing {}".format(os.path.basename(path))

    buffer = []          # token ids, stream positions [stream_pos, stream_pos+len)
    stream_pos = 0       # absolute index of buffer[0]
    next_p = 0           # next window start position to emit

    def cb(read, total_bytes=total_bytes):
        nonlocal last_pct
        if progress and total_bytes > 0:
            pct = int(read * 100 // total_bytes)
            if pct > last_pct:
                sys.stderr.write("\r{}: {}% ({:.0f} / {:.0f} MB)".format(
                    label, pct, read / 1e6, total_bytes / 1e6))
                sys.stderr.flush()
                last_pct = pct

    for chunk in _stream(path, tokenizer, chunk_size, cb):
        buffer.extend(chunk)
        # emit every window start p whose last target token (p+seq_len) exists
        while next_p + seq_len < stream_pos + len(buffer):
            off = next_p - stream_pos
            inputs = np.asarray(buffer[off:off + seq_len], dtype=np.int64)
            targets = np.asarray(buffer[off + 1:off + seq_len + 1], dtype=np.int64)
            yield inputs, targets
            next_p += stride
            drop = next_p - stream_pos
            if drop > 0:
                del buffer[:drop]
                stream_pos += drop
    if progress and total_bytes > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()


def iter_batches(path, tokenizer, seq_len, stride, batch_size,
                 chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
    """Yield (inputs, targets) numpy arrays of shape (batch_size, seq_len).

    Batch b = windows [b*batch_size, (b+1)*batch_size), exactly the C++
    slice training_data[b*total_input_rows : (b+1)*total_input_rows] reshaped.
    """
    windows = iter_windows(path, tokenizer, seq_len, stride,
                           chunk_size=chunk_size, progress=progress)
    inputs = []
    targets = []
    for inp, tgt in windows:
        inputs.append(inp)
        targets.append(tgt)
        if len(inputs) == batch_size:
            yield np.stack(inputs), np.stack(targets)
            inputs = []
            targets = []
    if inputs:
        sys.stderr.write("note: {} trailing window(s) dropped "
                         "(not a full batch of {})\n".format(len(inputs), batch_size))
        sys.stderr.flush()
