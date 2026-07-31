"""Degeneracy filter for generated replay targets (PLAN §4).

"Degeneracy filter only — no quality gating (except run B2). Exact rules, do not
improvise: drop if stripped output is empty; drop if < 3 whitespace tokens; drop
if the whole output is a single line under 8 characters; drop if any 10-gram
repeats more than 4 times."

The rules are transcribed literally; ``degeneracy_reason`` returns the first one
that fires so the manifest can report a per-reason breakdown.
"""

from __future__ import annotations

from collections import Counter

REPEAT_NGRAM = 10
REPEAT_MAX = 4          # "repeats more than 4 times" -> drop at 5+ occurrences
MIN_TOKENS = 3
SINGLE_LINE_MIN_CHARS = 8

REASONS = ("empty", "too_few_tokens", "single_short_line", "repeated_10gram")


def degeneracy_reason(text: str) -> str | None:
    """``None`` if the output is usable, else the name of the rule that dropped it."""
    s = (text or "").strip()
    if not s:
        return "empty"

    toks = s.split()
    if len(toks) < MIN_TOKENS:
        return "too_few_tokens"

    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) == 1 and len(s) < SINGLE_LINE_MIN_CHARS:
        return "single_short_line"

    if len(toks) >= REPEAT_NGRAM:
        counts = Counter(
            tuple(toks[i:i + REPEAT_NGRAM]) for i in range(len(toks) - REPEAT_NGRAM + 1)
        )
        if max(counts.values()) > REPEAT_MAX:
            return "repeated_10gram"

    return None


def is_degenerate(text: str) -> bool:
    return degeneracy_reason(text) is not None


def filter_outputs(texts):
    """Split an iterable of outputs into ``(kept_indices, reason_counter)``."""
    kept, reasons = [], Counter()
    for i, t in enumerate(texts):
        r = degeneracy_reason(t)
        if r is None:
            kept.append(i)
        else:
            reasons[r] += 1
    return kept, reasons
