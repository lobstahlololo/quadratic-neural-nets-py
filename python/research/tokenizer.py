"""Streaming, memory-bounded character tokenizer for large corpora.

The research pipeline uses a character-level vocabulary (same semantics as the
original tokenizer: the sorted set of unique characters in the corpus). The
original implementation called ``Path(path).read_text()``, which loads the
entire corpus into memory as one giant string and then builds a ``set`` over
every character. For a ~1.9 GB TinyStories train file that means holding
multiple GB in memory and iterating ~1.9 billion characters before a single
token can be produced, which looks like a hang.

This version builds the vocabulary by streaming the file in chunks (text
mode, so UTF-8 decoding and universal-newline translation are handled by the
C-level TextIOWrapper exactly as ``read_text()`` did), updating the character
set in C-speed ``set.update`` calls, and reporting progress to stderr so
construction is visibly progressing. Memory use is bounded by the chunk size,
not the corpus size. The resulting vocabulary is identical to the old one
(chunking never changes the character set or ordering), so the tokenizer
remains deterministic and API-compatible.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Read 8 MiB of characters per chunk: large enough for sequential-read
# throughput, small enough that each chunk is a few million characters and
# the working set stays tiny.
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def _report_progress(read_bytes, total_bytes, progress, last_pct, label):
    """Overwrite a single stderr line with current progress; return new pct."""
    if not progress or total_bytes <= 0:
        return last_pct
    pct = int(read_bytes * 100 // total_bytes)
    if pct > last_pct:
        sys.stderr.write(
            "\r{}: {}% ({:.0f} / {:.0f} MB)".format(
                label, pct, read_bytes / 1e6, total_bytes / 1e6
            )
        )
        sys.stderr.flush()
        return pct
    return last_pct


class CharacterTokenizer:
    def __init__(self, text: str):
        # Constructor for small in-memory texts (kept for API compatibility).
        # For large corpora prefer CharacterTokenizer.from_file() /
        # build_tokenizer(), which stream the corpus instead of holding it in
        # memory.
        characters = sorted(set(text))
        self.token_to_id = {
            character: index for index, character in enumerate(characters)
        }
        self.id_to_token = {
            index: character for character, index in self.token_to_id.items()
        }

    @classmethod
    def from_file(cls, path, chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
        """Build the vocabulary by streaming the file in chunks.

        Deterministic: the resulting vocabulary is exactly the sorted set of
        unique characters in the file, identical to what a whole-file
        ``read_text()`` read would produce (same UTF-8 decoding and same
        universal-newline translation).
        """
        total_bytes = Path(path).stat().st_size
        characters = set()
        read_bytes = 0
        last_pct = -1
        label = "building vocabulary"
        with open(path, "r", encoding="utf-8") as source:
            while True:
                text = source.read(chunk_size)
                if not text:
                    break
                characters.update(text)
                read_bytes = source.buffer.tell()
                last_pct = _report_progress(read_bytes, total_bytes, progress, last_pct, label)
        if progress and total_bytes > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = {
            character: index for index, character in enumerate(sorted(characters))
        }
        tokenizer.id_to_token = {
            index: character for character, index in tokenizer.token_to_id.items()
        }
        return tokenizer

    @property
    def vocabulary_size(self):
        return len(self.token_to_id)

    def encode(self, text: str):
        return [self.token_to_id[character] for character in text]

    def decode(self, token_ids):
        return "".join(self.id_to_token[token_id] for token_id in token_ids)

    def iter_file_tokens(self, path, chunk_size=DEFAULT_CHUNK_SIZE, progress=True,
                         progress_cb=None):
        """Stream a file, yielding lists of token ids (one list per chunk).

        Memory-bounded: never materializes the whole corpus. Use this for
        large files (e.g. the ~1.9 GB train corpus); a flat list of ids for
        that file would be billions of Python ints.

        ``progress_cb(read_bytes, total_bytes)``, if given, is called after
        each chunk with the exact byte position (replacing the built-in
        progress line; the built-in line is shown only when progress_cb is
        None).
        """
        total_bytes = Path(path).stat().st_size
        read_bytes = 0
        last_pct = -1
        label = "tokenizing {}".format(Path(path).name)
        with open(path, "r", encoding="utf-8") as source:
            while True:
                text = source.read(chunk_size)
                if not text:
                    break
                yield [self.token_to_id[character] for character in text]
                read_bytes = source.buffer.tell()
                if progress_cb is not None:
                    progress_cb(read_bytes, total_bytes)
                else:
                    last_pct = _report_progress(read_bytes, total_bytes, progress, last_pct, label)
        if progress_cb is None and progress and total_bytes > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def encode_file(self, path, chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
        """Tokenize a whole file into a flat list of token ids.

        Convenience wrapper over iter_file_tokens(). Appropriate for files up
        to a few hundred MB (the ~19 MB validation file is fine); for the
        multi-GB train corpus use iter_file_tokens() and consume per chunk.
        """
        ids = []
        for chunk in self.iter_file_tokens(path, chunk_size=chunk_size, progress=progress):
            ids.extend(chunk)
        return ids


def build_tokenizer(path, chunk_size=DEFAULT_CHUNK_SIZE, progress=True):
    """Build a CharacterTokenizer from a corpus file, streaming it in chunks."""
    return CharacterTokenizer.from_file(path, chunk_size=chunk_size, progress=progress)
