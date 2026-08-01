"""Tülu-3 gold replay slot — A1's reference locus, and A4's gold half (PLAN §9).

Reuses ``load_general_examples`` from ``socrateach_sft/prepare_socrateach_sft.py``
verbatim (same parquet shard, same English filter, same seeded ordering) so A1 is
the *same* replay stream Impl 2 trained on, not a re-implementation of it.

The ordering that function produces is seed-deterministic, so A4's 3,748 Tülu
examples are exactly the first half of A1's 7,496 — the two arms share a prefix
rather than sampling independently.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

from .paths import PREPARE_SOCRATEACH_PY

TULU_ID = "allenai/tulu-3-sft-olmo-2-mixture-0225"
KIND = "general_gold_tulu"


@lru_cache(maxsize=1)
def _prepare_module():
    spec = importlib.util.spec_from_file_location(
        "impl4_prepare_socrateach_ref", PREPARE_SOCRATEACH_PY
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {PREPARE_SOCRATEACH_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tulu_slot(n: int, seed: int) -> list[dict]:
    """``n`` SI-free English Tülu-3 conversations in Impl 4's record schema."""
    if n <= 0:
        return []
    recs = _prepare_module().load_general_examples(n, seed)
    if len(recs) < n:
        raise RuntimeError(
            f"Tulu slot short: asked for {n}, got {len(recs)}. The single parquet shard "
            f"load_general_examples() reads may not hold enough English conversations; "
            f"widen it in prepare_socrateach_sft.py rather than silently under-filling."
        )
    out = []
    for r in recs[:n]:
        assert all(m["role"] != "system" for m in r["messages"]), \
            "Tulu replay records must be SI-free (PLAN §11 check 3)"
        out.append({
            "messages": r["messages"],
            "problem_id": None,
            "dialogue_id": r["dialogue_id"],
            "answer": None,
            "source": TULU_ID,
            # Coarse stream label matching Impl 3's tagging; Impl 4's provenance in replay_kind.
            "kind": "general",
            "replay_kind": KIND,
            "superni_task_id": None,
            "sample_T": None,
            "sample_top_k": None,
            "sample_top_p": None,
            "gate_passed": None,
        })
    return out
