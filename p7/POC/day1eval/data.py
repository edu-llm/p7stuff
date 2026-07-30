"""MRBench loading + dialogue parsing.

Two jobs:
  1. Fetch/cache the MRBench json splits from GitHub.
  2. Turn each item's free-text ``conversation_history`` into structured turns.

The conversation_history is a single string where turns are introduced by
``Tutor:`` / ``Student:`` labels. Turns can themselves contain newlines (a
student's multi-line worked solution), and labels are often preceded by a
non-breaking space (U+00A0), so we split on the *label*, not on newlines.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field

from config import DATA_DIR, DATASETS


# Our canonical dimension key -> the key MRBench uses in its human annotations.
# Identical except MRBench spells humanlikeness in lowercase.
HUMAN_DIM_KEYS: dict[str, str] = {
    "Mistake_Identification": "Mistake_Identification",
    "Mistake_Location": "Mistake_Location",
    "Revealing_of_the_Answer": "Revealing_of_the_Answer",
    "Providing_Guidance": "Providing_Guidance",
    "Actionability": "Actionability",
    "Coherence": "Coherence",
    "Tutor_Tone": "Tutor_Tone",
    "Humanlikeness": "humanlikeness",
}


# A turn starts at string start OR after a newline (+ optional whitespace incl.
# the U+00A0 the dataset uses), then "Tutor:" / "Student:". \s matches U+00A0.
_TURN_RE = re.compile(r"(?:^|\n)\s*(Tutor|Student)\s*:\s*", re.IGNORECASE)


@dataclass
class Turn:
    role: str   # "Tutor" or "Student"
    text: str


@dataclass
class Dialogue:
    conversation_id: str
    turns: list[Turn]
    raw_history: str
    ground_truth_solution: str = ""
    data: str = ""      # "MathDial" or "Bridge"
    split: str = ""
    topic: str = ""

    @property
    def last_student_turn(self) -> str:
        for t in reversed(self.turns):
            if t.role == "Student":
                return t.text
        return ""


@dataclass
class AnnotatedResponse:
    """One MRBench human-annotated tutor response (the gold labels MRBench ships).

    Used by ``validate_judge.py`` to measure how well the LLM judge agrees with
    expert humans (per-dimension Cohen's kappa) BEFORE trusting any generated
    tutor score.
    """

    conversation_id: str
    tutor_name: str
    raw_history: str
    ground_truth_solution: str
    response: str
    human: dict[str, str | None] = field(default_factory=dict)  # canonical_dim_key -> label


def _cache_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"mrbench_{dataset.lower()}.json")


def download(dataset: str, force: bool = False) -> str:
    """Download one dataset split to the local cache; return its path."""
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Choose from {list(DATASETS)}.")
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _cache_path(dataset)
    if force or not os.path.exists(path):
        url = DATASETS[dataset]
        print(f"[data] downloading {dataset} <- {url}")
        urllib.request.urlretrieve(url, path)
    return path


def parse_history(history: str) -> list[Turn]:
    """Split a raw conversation_history string into ordered Turn objects."""
    history = history.replace("\xa0", " ")
    matches = list(_TURN_RE.finditer(history))
    turns: list[Turn] = []
    for i, m in enumerate(matches):
        role = m.group(1).capitalize()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(history)
        text = history[start:end].strip()
        if text:
            turns.append(Turn(role=role, text=text))
    return turns


def load_dialogues(dataset: str, limit: int = 0, force_download: bool = False) -> list[Dialogue]:
    """Load a split and return parsed Dialogue objects (``limit`` 0 = all)."""
    path = download(dataset, force=force_download)
    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    dialogues: list[Dialogue] = []
    for idx, item in enumerate(items):
        history = item.get("conversation_history", "")
        turns = parse_history(history)
        if not turns:
            continue
        dialogues.append(
            Dialogue(
                conversation_id=str(item.get("conversation_id", idx)),
                turns=turns,
                raw_history=history,
                ground_truth_solution=item.get("Ground_Truth_Solution", ""),
                data=item.get("Data", ""),
                split=item.get("Split", ""),
                topic=item.get("Topic", ""),
            )
        )
        if limit and len(dialogues) >= limit:
            break
    return dialogues


def load_annotated_responses(
    dataset: str = "V1",
    limit: int = 0,
    seed: int = 0,
    force_download: bool = False,
) -> list[AnnotatedResponse]:
    """Load MRBench's human-annotated tutor responses (``anno_llm_responses``).

    Each MRBench item carries several candidate tutors (Expert, GPT4, Gemini,
    Llama, ...), each with a human label on all 8 dimensions. We flatten these
    into one row per (conversation, tutor). ``limit`` samples deterministically
    (shuffled by ``seed``) so a quick validation run is still representative.
    """
    path = download(dataset, force=force_download)
    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    rows: list[AnnotatedResponse] = []
    for item in items:
        history = item.get("conversation_history", "").replace("\xa0", " ")
        solution = item.get("Ground_Truth_Solution", "")
        for tutor_name, entry in item.get("anno_llm_responses", {}).items():
            annotation = entry.get("annotation", {}) or {}
            response = entry.get("response", "")
            if not response or not annotation:
                continue
            rows.append(
                AnnotatedResponse(
                    conversation_id=str(item.get("conversation_id", "")),
                    tutor_name=tutor_name,
                    raw_history=history,
                    ground_truth_solution=solution,
                    response=response,
                    human={dk: annotation.get(hk) for dk, hk in HUMAN_DIM_KEYS.items()},
                )
            )
    random.Random(seed).shuffle(rows)  # deterministic sample
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    # Quick smoke test: parse the default split and show turn stats.
    import argparse

    from config import DEFAULT_DATASET

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    ds = load_dialogues(args.dataset, limit=args.limit)
    print(f"[data] parsed {len(ds)} dialogues from {args.dataset}")
    for d in ds:
        roles = " -> ".join(t.role[0] for t in d.turns)
        print(f"\n== {d.conversation_id} | {d.data} | turns: {len(d.turns)} [{roles}]")
        print(f"   last student: {d.last_student_turn[:120]!r}")
