"""Impl 5 — self-distilled *pedagogy targets* for low-KL SFT.

Impl 4 self-distilled the replay slot and pinned δ = 0. Impl 5 is the other half: the tutor
turns themselves are rewritten in π₀'s own words, and the replay slot goes back to Tülu-3
gold. The two are orthogonal and composable; the δ=1 × σ=1 cell is out of scope.

The library deliberately depends on ``../impl4_ssd`` (see :mod:`impl5._impl4`) so that the
tokenisation, degeneracy rules and token-matching are the *same objects* both projects use.
"""

from __future__ import annotations

__all__ = ["answer_leak", "chat5", "config5", "dialogue", "distill", "gate5", "paths5"]
