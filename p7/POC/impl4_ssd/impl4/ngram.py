"""13-gram decontamination against the eval team's prompt sets (PLAN §3, filter 2).

``day1eval/decontam.py`` no longer exists in the tree, so this is written fresh.

Contaminating ``math_eval/`` or ``general_eval/`` would invalidate the whole
experiment, so the check errs toward over-dropping:

* references shorter than ``n`` tokens (many ``general_prompts.jsonl`` entries are
  6-10 words) contribute an **exact contiguous-phrase** rule instead of an n-gram,
  because a 13-gram index simply cannot see them;
* candidate text is the concatenation of everything we would train on for that
  instance (definition + input + gold output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .textutil import norm_tokens

DEFAULT_N = 13


class NGramIndex:
    """Membership test for "does this text share an n-gram with any reference?"."""

    def __init__(self, n: int = DEFAULT_N):
        self.n = n
        self._grams: set[tuple[str, ...]] = set()
        self._short: list[tuple[str, ...]] = []   # references with < n tokens
        self.n_refs = 0

    def add(self, text: str) -> None:
        toks = norm_tokens(text)
        if not toks:
            return
        self.n_refs += 1
        if len(toks) < self.n:
            self._short.append(tuple(toks))
            return
        for i in range(len(toks) - self.n + 1):
            self._grams.add(tuple(toks[i:i + self.n]))

    def add_all(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    def hit(self, text: str) -> tuple[str, ...] | None:
        """Return the offending gram/phrase, or ``None`` if the text is clean."""
        toks = norm_tokens(text)
        if not toks:
            return None
        if self._grams and len(toks) >= self.n:
            for i in range(len(toks) - self.n + 1):
                g = tuple(toks[i:i + self.n])
                if g in self._grams:
                    return g
        for phrase in self._short:
            k = len(phrase)
            if k > len(toks):
                continue
            for i in range(len(toks) - k + 1):
                if tuple(toks[i:i + k]) == phrase:
                    return phrase
        return None

    def __len__(self) -> int:
        return len(self._grams) + len(self._short)


def load_eval_prompts(path: str | Path, fields: tuple[str, ...] = ("prompt",)) -> list[str]:
    """Pull the prompt text out of a JSONL eval file."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in fields:
                v = row.get(field)
                if isinstance(v, str) and v.strip():
                    out.append(v)
    return out


def build_eval_index(paths: Iterable[str | Path], n: int = DEFAULT_N) -> NGramIndex:
    """Index every eval prompt we must not overlap with."""
    idx = NGramIndex(n=n)
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(
                f"decontamination target missing: {p}. These files are owned by the eval "
                f"team and must be present — do not skip the check."
            )
        idx.add_all(load_eval_prompts(p))
    return idx
