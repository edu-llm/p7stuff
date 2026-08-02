"""The pedagogy quality gate (PLAN §4): keep the rewrite, or fall back to the gold turn.

Four stages, first failure wins, so the manifest can attribute every fallback::

    0  degeneracy      impl4/degeneracy.py, verbatim, + "generation hit the token cap"
    1  answer leakage   conditional on gold — see answer_leak.py
    2  one step / one idea   length, question count, enumerated lists, sentence count
    3  intent match     ROUGE-L F1 against gold, impl4/gate.py:rouge_l_f1

**On failure it falls back to gold; it does not resample.** That is the opposite of impl4's
B2 gate, and deliberately so. There, falling back would have reinjected the off-policy
targets the arm existed to remove. Here gold fallback *is* the spec (SDFT Eq. 4), and its
cost is a lower realised δ — which is why realised δ is a first-class manifest field rather
than a diagnostic.

Stage 3's threshold is the one to be careful with, and its asymmetry is the reverse of B2's:
too **high** keeps only near-copies, which is vanilla SFT with extra steps and removes the
very KL reduction being bought; too **low** and the rewrite no longer sets up the gold
student reply that follows it, so the dialogue stops making sense.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ._impl4 import degeneracy, gate as gate4
from .answer_leak import leaks_conditional
from .config5 import DEFAULT_THRESHOLDS, GateThresholds

#: An enumerated step ("1. ", "2) ", "- ", "* ") at the start of a line.
_LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s")
#: Sentence terminators. Crude on purpose — this is a heuristic, not a parser.
_SENT = re.compile(r"[.!?]+(?:\s|$)")

#: "decontamination" is not a per-turn stage — it is a whole-dialogue revert applied
#: after the pass (see distill_pedagogy.py) — but it lands in the same verdict stream,
#: so it belongs in the summary or its fallbacks vanish from by_stage.
STAGES = ("degeneracy", "answer_leak", "one_step", "intent_match", "decontamination")


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str                        # "ok", or the rule that fired
    stage: str | None                  # which of STAGES rejected it
    rouge: float

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason, "stage": self.stage,
                "rouge": round(self.rouge, 4)}


def _list_lines(text: str) -> int:
    return sum(1 for ln in (text or "").splitlines() if _LIST_LINE.match(ln))


def _sentences(text: str) -> int:
    return len([s for s in _SENT.split(text or "") if s.strip()])


def one_step_reason(rewrite: str, gold: str, th: GateThresholds) -> str | None:
    """Stage 2. ``None`` if the rewrite still reads as a single move."""
    w_new, w_gold = len((rewrite or "").split()), len((gold or "").split())
    if w_new > max(th.word_ratio * w_gold, th.word_floor):
        return "too_long"
    if (rewrite or "").count("?") > th.max_questions:
        return "too_many_questions"
    if _list_lines(rewrite) >= th.list_min_lines and _list_lines(gold) == 0:
        return "enumerated_list"
    if _sentences(rewrite) > th.max_sentences:
        return "too_many_sentences"
    return None


def evaluate(rewrite: str, gold: str, answer, finished: bool = True,
             th: GateThresholds = DEFAULT_THRESHOLDS) -> Verdict:
    """Run the four stages in order and return the first failure, or a pass.

    ``finished=False`` means generation stopped at ``max_new_tokens`` without emitting EOS.
    That is a truncated sentence, not a tutor turn, so it is rejected at stage 0. Stage 2's
    length rule would catch most such cases anyway, but not one whose gold turn is long —
    and a mid-word cut is exactly the kind of target that would teach the model to stop
    mid-word.
    """
    if not finished:
        return Verdict(False, "unterminated", "degeneracy", 0.0)
    d = degeneracy.degeneracy_reason(rewrite)
    if d is not None:
        return Verdict(False, f"degenerate_{d}", "degeneracy", 0.0)

    leak = leaks_conditional(rewrite, gold, answer)
    if leak is not None:
        return Verdict(False, leak, "answer_leak", 0.0)

    step = one_step_reason(rewrite, gold, th)
    if step is not None:
        return Verdict(False, step, "one_step", 0.0)

    # ROUGE-L is only computed once the cheap rules have passed, so the per-turn cost of
    # the gate on a rejected rewrite is a few string ops rather than an O(n·m) LCS.
    r = gate4.rouge_l_f1(rewrite, gold)
    if r < th.rouge_min:
        return Verdict(False, "low_rouge", "intent_match", r)
    return Verdict(True, "ok", None, r)


def summarize(verdicts) -> dict:
    """Per-stage and per-reason fallback rates over an iterable of :class:`Verdict`."""
    v = list(verdicts)
    n = len(v)
    stages = Counter(x.stage for x in v if not x.passed)
    reasons = Counter(x.reason for x in v if not x.passed)
    kept = [x for x in v if x.passed]
    return {
        "n": n,
        "n_kept": len(kept),
        "keep_rate": round(len(kept) / n, 4) if n else None,
        "fallback_rate": round((n - len(kept)) / n, 4) if n else None,
        "by_stage": {s: stages.get(s, 0) for s in STAGES},
        "by_stage_rate": {s: round(stages.get(s, 0) / n, 4) for s in STAGES} if n else {},
        "by_reason": dict(reasons.most_common()),
        "mean_rouge_kept": round(sum(x.rouge for x in kept) / len(kept), 4) if kept else None,
    }
