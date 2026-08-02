"""Split a SocraTeach record into turns, and put it back together with rewritten targets.

A pedagogy record is a strict alternation

    system(SI), user(problem), assistant(t₁), user(s₁), assistant(t₂), …, assistant(t_N)

so ``N`` tutor turns and ``N-1`` student turns, always ending on the tutor. The rewriting
pass (PLAN §3.1) walks ``r = 1…N`` and at each round conditions on the **already-rewritten**
prefix, which is what makes the generation context equal the training context.

Nothing here mutates the source record: :func:`with_rewritten` returns a new dict carrying
the same ``dialogue_id`` / ``problem_id`` / ``answer`` / ``source`` / ``kind``, so a
distilled pool diffs cleanly against the gold pool row-for-row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config5 import TEMPLATE_DEFAULT, reference_block


@dataclass
class Dialogue:
    """One SocraTeach dialogue, decomposed."""

    record: dict
    system: str
    problem: str
    tutor: list[str]                   # gold t₁ … t_N
    student: list[str]                 # s₁ … s_{N-1}

    @property
    def dialogue_id(self) -> str:
        return self.record["dialogue_id"]

    @property
    def answer(self):
        return self.record.get("answer")

    @property
    def n_turns(self) -> int:
        return len(self.tutor)

    # -- prompts ------------------------------------------------------------
    def training_messages(self, rewritten: Sequence[str], upto: int) -> list[dict]:
        """The **training** prefix for round ``upto`` (1-indexed): no reference block.

        ``rewritten[k]`` is the accepted target for turn ``k+1``; only the first
        ``upto-1`` are consulted, so a partly-complete rewrite list is fine.
        """
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.problem}]
        for k in range(upto - 1):
            msgs.append({"role": "assistant", "content": rewritten[k]})
            msgs.append({"role": "user", "content": self.student[k]})
        return msgs

    def distill_messages(self, rewritten: Sequence[str], upto: int,
                         template: str = TEMPLATE_DEFAULT) -> list[dict]:
        """The **distillation** prompt for round ``upto``: training prefix + reference.

        PLAN §3.2 — the reference is appended to the content of the *last user message*
        rather than inserted as a new turn, so role alternation matches training exactly
        and the divergence is a checkable suffix.
        """
        msgs = self.training_messages(rewritten, upto)
        tail = dict(msgs[-1])
        assert tail["role"] == "user", "the distillation prompt must end on a user turn"
        tail["content"] = tail["content"] + reference_block(self.tutor[upto - 1], template)
        msgs[-1] = tail
        return msgs

    # -- output -------------------------------------------------------------
    def with_rewritten(self, rewritten: Sequence[str]) -> dict:
        """A pool record whose assistant turns are ``rewritten``, everything else unchanged."""
        if len(rewritten) != self.n_turns:
            raise ValueError(f"{self.dialogue_id}: {len(rewritten)} rewrites for "
                             f"{self.n_turns} tutor turns")
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.problem}]
        for k, t in enumerate(rewritten):
            msgs.append({"role": "assistant", "content": t})
            if k < len(self.student):
                msgs.append({"role": "user", "content": self.student[k]})
        out = {k: self.record.get(k) for k in
               ("problem_id", "dialogue_id", "answer", "source", "kind")}
        out["messages"] = msgs
        return {"messages": msgs, **{k: v for k, v in out.items() if k != "messages"}}


def parse(record: dict) -> Dialogue:
    """Decompose one pedagogy record; raises if it is not the expected alternation."""
    msgs = record["messages"]
    did = record.get("dialogue_id")
    if not msgs or msgs[0]["role"] != "system":
        raise ValueError(f"{did}: pedagogy record must start with a system message")
    if len(msgs) < 3 or msgs[1]["role"] != "user":
        raise ValueError(f"{did}: expected system, user(problem), assistant, …")

    tutor, student = [], []
    expect = "assistant"
    for m in msgs[2:]:
        if m["role"] != expect:
            raise ValueError(f"{did}: expected {expect}, got {m['role']} — "
                             f"the rewriting pass assumes strict alternation")
        (tutor if expect == "assistant" else student).append(m["content"])
        expect = "user" if expect == "assistant" else "assistant"
    if expect != "user":
        raise ValueError(f"{did}: dialogue does not end on a tutor turn")
    return Dialogue(record=record, system=msgs[0]["content"], problem=msgs[1]["content"],
                    tutor=tutor, student=student)


def parse_all(records: Iterable[dict]) -> list[Dialogue]:
    return [parse(r) for r in records]


def round_schedule(dialogues: Sequence[Dialogue]) -> dict[int, int]:
    """``{round: how many dialogues participate}``. Round ``r`` runs every ``N >= r``."""
    out: dict[int, int] = {}
    for d in dialogues:
        for r in range(1, d.n_turns + 1):
            out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items()))
