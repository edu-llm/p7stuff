"""B2's quality gate (PLAN §9) — the only gating experiment in Impl 4.

"Keep a sample if the normalized gold string appears as a substring of the output
**or** ROUGE-L F1 >= 0.3. On failure, resample (up to 4 tries), do not fall back
to gold — falling back reinjects off-policy targets exactly where we're trying to
remove them."

ROUGE-L is implemented here (LCS over normalised whitespace tokens) rather than
pulled from ``rouge-score``: it is ~30 lines, it removes a cluster-install
dependency, and it lets the normalisation match :mod:`impl4.textutil` exactly.
"""

from __future__ import annotations

from .textutil import normalize, norm_tokens

GATE_THRESHOLD = 0.3
MAX_TRIES = 4


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, O(len(a)·len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    """ROUGE-L F1 (β=1) over normalised whitespace tokens."""
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p or not g:
        return 0.0
    l = lcs_length(p, g)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(g)
    return 2 * prec * rec / (prec + rec)


def gate_result(pred: str, gold: str, threshold: float = GATE_THRESHOLD) -> tuple[bool, str, float]:
    """``(passed, how, rouge_l_f1)``.

    ``how`` is ``"substring"``, ``"rouge_l"`` or ``"fail"`` so the manifest can
    report which arm of the disjunction is carrying the gate.
    """
    ng = normalize(gold)
    score = rouge_l_f1(pred, gold)
    if ng and ng in normalize(pred):
        return True, "substring", score
    if score >= threshold:
        return True, "rouge_l", score
    return False, "fail", score


def gate_passed(pred: str, gold: str, threshold: float = GATE_THRESHOLD) -> bool:
    return gate_result(pred, gold, threshold)[0]
