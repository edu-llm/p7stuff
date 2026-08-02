"""Stage 1 of the gate: does the rewrite give away the final answer? (PLAN §4 Stage 1)

**Conditional on gold, and that is the whole point.** A good tutor does not hand over the
answer — except at the end, after the student has produced it. Measured on the published
pool, roughly half of *final* tutor turns legitimately state the answer. An unconditional
"does the rewrite state the answer?" rule would therefore fall back to gold on half of all
final turns — the highest-KL, most behavioural turns in the dataset — and Impl 5 would
quietly degenerate into Impl 2 while every log looked healthy.

So the rule is::

    fail  iff  leaks(rewrite) and not leaks(gold)

Modelled on ``math_eval/grade_math_logic.py``'s ``int`` branch (last integer literal, commas
stripped) rather than importing it: that script is CLI-shaped, executes at import time, and
is owned by the eval team. What is reused is the *normalisation*, not the code.
"""

from __future__ import annotations

import re

#: Integers and decimals, with thousands separators. Deliberately greedy about commas so
#: "1,200" reads as one literal rather than "1" and "200".
#:
#: The boundaries are load-bearing in both directions. Without the lookbehind, the "00" of
#: "25.00" reads as a second literal ``0`` and any alphanumeric token containing the answer's
#: digits ("step42", "w99") reads as a reveal — over-firing the leak rule, which costs
#: realised δ on turns that never leaked anything. Without the lookahead, "42nd" counts as
#: stating 42.
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w)")

#: Non-numeric reveals. Fired only when the gold turn contains none of them.
LEAK_PHRASES = ("the answer is", "so the answer", "the final answer")


def normalize_number(s: str) -> str | None:
    """``"1,200" -> "1200"``, ``"30.0" -> "30"``, ``"7.50" -> "7.5"``. ``None`` if unparseable.

    Trailing-zero normalisation matters because the pool stores answers as strings and mixes
    forms: ``"30.0"`` and ``"30"`` are the same answer and must compare equal, or the leak
    rule silently stops firing on every decimal-valued problem.
    """
    if s is None:
        return None
    t = str(s).strip().replace(",", "").rstrip(".")
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(v).rstrip("0").rstrip(".")


def numeric_literals(text: str) -> set[str]:
    """Every number in ``text``, normalised."""
    out = set()
    for m in _NUM.findall(text or ""):
        n = normalize_number(m)
        if n is not None:
            out.add(n)
    return out


def states_answer(text: str, answer) -> bool:
    """Does ``text`` contain the answer value among its numeric literals?"""
    a = normalize_number(answer)
    if a is None:                      # non-numeric answer: nothing to match on
        return False
    return a in numeric_literals(text)


def has_leak_phrase(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in LEAK_PHRASES)


def leaks_conditional(rewrite: str, gold: str, answer) -> str | None:
    """``None`` if the rewrite is acceptable, else which conditional rule fired.

    Both rules are conditional on gold: a rewrite is only penalised for revealing something
    the gold turn kept back.
    """
    if states_answer(rewrite, answer) and not states_answer(gold, answer):
        return "answer_leak_value"
    if has_leak_phrase(rewrite) and not has_leak_phrase(gold):
        return "answer_leak_phrase"
    return None
