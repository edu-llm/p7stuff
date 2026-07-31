"""Small text helpers shared by the filters. Stdlib only, so they unit-test locally."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
# Punctuation *and* symbols. Symbols matter: `+`, `=`, `*`, `$` are Unicode category
# S, not P, and `general_prompts.jsonl` is full of them ("Solve for x: 3x + 6 = 21.",
# "What is 17 * 23?"). Leaving them in would let a paraphrase of an eval prompt slip
# past the decontamination check, and over-matching is the safe direction here.
_STRIP_TABLE = {
    i: " " for i in range(0x110000)
    if unicodedata.category(chr(i))[0] in ("P", "S")
}


def normalize(text: str) -> str:
    """Lowercase, NFKC, punctuation/symbols → space, collapse whitespace.

    Used for n-gram decontamination and the B2 gate so that cosmetic differences
    (quotes, hyphenation, casing, math operators) neither hide contamination nor
    fail a gate.
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.translate(_STRIP_TABLE)
    return _WS.sub(" ", t).strip()


def norm_tokens(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def word_count(text: str) -> int:
    """Whitespace words, as PLAN §3 filter 3 specifies (no normalisation)."""
    return len((text or "").split())
