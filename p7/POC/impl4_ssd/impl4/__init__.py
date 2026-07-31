"""Impl 4 — self-distilled replay for low-KL pedagogy SFT.

Library code shared by the ``impl4_ssd/*.py`` entrypoints. See ``PLAN.md`` for the
build spec and ``RUNBOOK.md`` for the command sequence.

Scope reminder (PLAN.md header): this package builds data, trains, and saves
checkpoints. It contains no evaluation, KL, judge, or grading code.
"""

__all__ = [
    "chat",
    "config",
    "degeneracy",
    "gate",
    "generate",
    "manifest",
    "mixing",
    "ngram",
    "paths",
    "superni",
    "textutil",
    "tulu",
]
