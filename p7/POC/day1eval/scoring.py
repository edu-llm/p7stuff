"""MRBench pedagogical rubric + LLM-as-a-judge prompt/parse/aggregate.

The 8 dimensions and their label sets mirror the MRBench annotation schema
(Maurya et al., NAACL 2025). Each dimension carries a numeric ``score`` map
(0..1, higher = pedagogically better) so we can report a mean per dimension.

Note on ``Revealing_of_the_Answer``: a *good* tutor does NOT reveal the final
answer, so "No" is the high-scoring label there.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dimension:
    key: str
    question: str
    labels: tuple[str, ...]
    score: dict[str, float]


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "Mistake_Identification",
        "Has the tutor identified/recognized that there is a mistake in the student's response?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
    Dimension(
        "Mistake_Location",
        "Does the tutor's response accurately point to the location of the mistake?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
    Dimension(
        "Revealing_of_the_Answer",
        "Does the tutor reveal the final answer? (A good tutor does NOT reveal it.)",
        # Full MRBench human-annotation labels, so judge output is directly
        # comparable to the ground-truth labels in validate_judge.py.
        ("Yes (and the answer is correct)", "Yes (but the answer is incorrect)", "No"),
        {"No": 1.0, "Yes (and the answer is correct)": 0.0, "Yes (but the answer is incorrect)": 0.0},
    ),
    Dimension(
        "Providing_Guidance",
        "Does the tutor offer correct and relevant guidance (explanation, hint, question, example)?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
    Dimension(
        "Actionability",
        "Is it clear from the tutor's feedback what the student should do next?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
    Dimension(
        "Coherence",
        "Is the tutor's response coherent and logically consistent with the conversation?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
    Dimension(
        "Tutor_Tone",
        "What is the tone of the tutor's response?",
        ("Encouraging", "Neutral", "Offensive"),
        {"Encouraging": 1.0, "Neutral": 0.5, "Offensive": 0.0},
    ),
    Dimension(
        "Humanlikeness",
        "Does the tutor's response sound natural/human rather than robotic or artificial?",
        ("Yes", "To some extent", "No"),
        {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    ),
)

DIM_BY_KEY = {d.key: d for d in DIMENSIONS}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI math tutors. You assess the pedagogical "
    "quality of a single tutor response to a student who has just made a mistake, "
    "following the MRBench rubric. Be strict and objective. Respond with JSON only."
)

# Tolerant aliases for judge outputs that use an older/shorter label spelling
# than the canonical MRBench label. Keys are lowercased; values must be an exact
# canonical label of some dimension.
_LABEL_ALIASES: dict[str, str] = {
    "yes (correct)": "Yes (and the answer is correct)",
    "yes (incorrect)": "Yes (but the answer is incorrect)",
    "yes, correct": "Yes (and the answer is correct)",
    "yes, incorrect": "Yes (but the answer is incorrect)",
}


def _rubric_block() -> str:
    lines = []
    for i, d in enumerate(DIMENSIONS, 1):
        opts = " / ".join(f'"{l}"' for l in d.labels)
        lines.append(f'{i}. {d.key}: {d.question}\n   Allowed values: {opts}')
    return "\n".join(lines)


def build_judge_messages(
    conversation_history: str,
    tutor_response: str,
    solution: str = "",
) -> list[dict[str, str]]:
    """Build chat messages asking the judge to rate one tutor response.

    ``solution`` is the reference (ground-truth) worked solution. When provided,
    it is shown to the judge so it can actually verify mistake-location, guidance
    correctness, and answer-revealing — LLM judges under-catch arithmetic errors
    without it. Real math tutors in MathDial had the solution too.
    """
    keys = ", ".join(f'"{d.key}"' for d in DIMENSIONS)
    sol_block = ""
    if solution and str(solution).strip():
        sol_block = (
            "=== Reference solution (ground truth; the tutor should GUIDE toward "
            "this, NOT simply reveal it) ===\n"
            f"{str(solution).strip()}\n\n"
        )
    user = (
        "Evaluate the tutor's response below against the rubric.\n\n"
        "=== Conversation so far ===\n"
        f"{conversation_history.strip()}\n\n"
        f"{sol_block}"
        "=== Tutor response to evaluate ===\n"
        f"{tutor_response.strip()}\n\n"
        "=== Rubric (choose exactly one allowed value per dimension) ===\n"
        f"{_rubric_block()}\n\n"
        "Return ONLY a JSON object with exactly these keys "
        f"({keys}), each mapped to one allowed value string. No prose, no markdown."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a possibly-fenced/prose-wrapped string."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")


def _canonical_label(dim: Dimension, value: Any) -> str | None:
    """Map a raw judge value to an allowed label (tolerant of case/whitespace)."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    for label in dim.labels:
        if v == label:
            return label
    low = v.lower()
    for label in dim.labels:
        if low == label.lower():
            return label
    # Tolerate known alternate spellings (e.g. older short "Yes (correct)" form).
    aliased = _LABEL_ALIASES.get(low)
    if aliased is not None and aliased in dim.labels:
        return aliased
    return None


def canonical_label(dim: Dimension, value: Any) -> str | None:
    """Public wrapper: map any raw label string to a canonical dimension label.

    Handy for canonicalising MRBench's human annotations before comparing them
    to judge output (see validate_judge.py).
    """
    return _canonical_label(dim, value)


def parse_judgment(text: str) -> dict[str, str]:
    """Parse judge output into {dimension_key: canonical_label}.

    Raises ValueError if no JSON is found. Missing/invalid per-dimension values
    are recorded as ``None`` rather than failing the whole record.
    """
    raw = _extract_json(text)
    result: dict[str, str | None] = {}
    for d in DIMENSIONS:
        result[d.key] = _canonical_label(d, raw.get(d.key))
    return result  # type: ignore[return-value]


def aggregate(judgments: list[dict[str, str | None]]) -> dict[str, Any]:
    """Compute per-dimension label distributions and mean numeric scores."""
    summary: dict[str, Any] = {}
    for d in DIMENSIONS:
        labels = [j.get(d.key) for j in judgments if j and j.get(d.key) in d.score]
        counts = Counter(j.get(d.key) for j in judgments if j)
        n_valid = len(labels)
        mean = sum(d.score[l] for l in labels) / n_valid if n_valid else None
        summary[d.key] = {
            "mean_score": round(mean, 4) if mean is not None else None,
            "n_valid": n_valid,
            "distribution": {l: counts.get(l, 0) for l in d.labels},
            "invalid_or_missing": sum(1 for j in judgments if not j or j.get(d.key) not in d.score),
        }
    valid_means = [v["mean_score"] for v in summary.values() if v["mean_score"] is not None]
    summary["_overall_mean_score"] = round(sum(valid_means) / len(valid_means), 4) if valid_means else None
    return summary


def format_summary(summary: dict[str, Any], title: str = "") -> str:
    """Pretty ASCII table of the aggregate summary."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'Dimension':<26}{'mean':>7}{'n':>6}   distribution")
    lines.append("-" * 72)
    for d in DIMENSIONS:
        s = summary[d.key]
        mean = "  n/a" if s["mean_score"] is None else f"{s['mean_score']:.3f}"
        dist = ", ".join(f"{k}:{v}" for k, v in s["distribution"].items())
        lines.append(f"{d.key:<26}{mean:>7}{s['n_valid']:>6}   {dist}")
    lines.append("-" * 72)
    overall = summary.get("_overall_mean_score")
    lines.append(f"{'OVERALL (mean of means)':<26}{'  n/a' if overall is None else f'{overall:.3f}':>7}")
    return "\n".join(lines)
