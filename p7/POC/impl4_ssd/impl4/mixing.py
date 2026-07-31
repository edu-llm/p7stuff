"""Token-matched subsampling (PLAN §5) and 24/8 block ordering (PLAN §6).

Why both, when either alone would do under *one* loss normalisation:

* ``transformers>=4.48`` normalises the loss by ``num_items_in_batch`` = total
  unmasked label tokens over the accumulation group, so stream weight is
  **token**-proportional. Token-matching every arm's replay slot to A1's token total
  is what keeps a ``T`` sweep from silently becoming a token-weight sweep.
* That fix depends on ``Trainer.model_accepts_loss_kwargs``, which can fall back to
  per-micro-batch mean when the model is PEFT-wrapped. Under *that* normalisation a
  fixed 24-pedagogy-then-8-general block gives the replay stream exactly 25% of every
  step's gradient, length-independently.

They cost nothing together, so we do both and let ``probe_loss_norm.py`` record
which normalisation actually fired.
"""

from __future__ import annotations

import bisect
import random
from collections import Counter
from typing import Sequence

from .config import (
    GEN_PER_BLOCK,
    PED_PER_BLOCK,
    SEED,
    TOKEN_MATCH_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------
def token_matched_select(
    counts: Sequence[int],
    n: int,
    target_total: int | None,
    seed: int = SEED,
    max_swaps: int = 4000,
) -> tuple[list[int], dict]:
    """Choose ``n`` of ``len(counts)`` candidates whose token total is closest to target.

    Shuffle first (so *content* is random), take a prefix, then greedily swap one
    in-set item for one out-of-set item at a time, each swap chosen to shrink
    ``|total - target|`` the most. Deterministic given ``seed``.

    ``target_total=None`` means "no matching required" — used for A1, which *is* the
    reference — and simply returns the shuffled prefix.
    """
    m = len(counts)
    if n > m:
        raise ValueError(f"need {n} candidates, only {m} available")

    idx = list(range(m))
    random.Random(seed).shuffle(idx)
    sel, rest = idx[:n], idx[n:]

    stats = {
        "n_candidates": m,
        "n_selected": n,
        "target_total": target_total,
        "swaps": 0,
    }
    total = sum(counts[i] for i in sel)
    stats["initial_total"] = total

    if target_total is None or not rest:
        stats["realized_total"] = total
        stats["ratio_to_target"] = None if target_total in (None, 0) else total / target_total
        return sorted(sel), stats

    # Sorted view of the out-of-set pool for O(log m) nearest-value lookups.
    rest_sorted = sorted(rest, key=lambda i: counts[i])
    rest_keys = [counts[i] for i in rest_sorted]

    swaps = 0
    while swaps < max_swaps:
        delta = target_total - total
        if delta == 0:
            break
        best = None  # (new_abs_delta, pos_in_sel, pos_in_rest)
        for si, i in enumerate(sel):
            want = counts[i] + delta
            p = bisect.bisect_left(rest_keys, want)
            for q in (p - 1, p):
                if 0 <= q < len(rest_keys):
                    nd = abs(delta - (rest_keys[q] - counts[i]))
                    if best is None or nd < best[0]:
                        best = (nd, si, q)
        if best is None or best[0] >= abs(delta):
            break
        _, si, q = best
        out_i, in_i = sel[si], rest_sorted[q]
        total += counts[in_i] - counts[out_i]
        sel[si] = in_i
        rest_sorted.pop(q)
        rest_keys.pop(q)
        pos = bisect.bisect_left(rest_keys, counts[out_i])
        rest_sorted.insert(pos, out_i)
        rest_keys.insert(pos, counts[out_i])
        swaps += 1

    stats["swaps"] = swaps
    stats["realized_total"] = total
    stats["ratio_to_target"] = total / target_total if target_total else None
    stats["within_tolerance"] = (
        target_total == 0 or abs(total - target_total) / target_total <= TOKEN_MATCH_TOLERANCE
    )
    return sorted(sel), stats


# ---------------------------------------------------------------------------
# Block ordering
# ---------------------------------------------------------------------------
def block_order(
    pedagogy: Sequence[dict],
    general: Sequence[dict],
    n_blocks: int,
    ped_per_block: int = PED_PER_BLOCK,
    gen_per_block: int = GEN_PER_BLOCK,
    seed: int = SEED,
) -> list[dict]:
    """Repeating blocks of ``[24 pedagogy, 8 general]``.

    Content is shuffled *within* each stream pool (seeded) so only the structure is
    fixed. Consumed by a ``SequentialSampler`` with ``per_device_batch=8``: positions
    0-7 / 8-15 / 16-23 are pedagogy micro-batches and 24-31 is the general one.
    """
    need_p, need_g = n_blocks * ped_per_block, n_blocks * gen_per_block
    if len(pedagogy) < need_p:
        raise ValueError(f"pedagogy pool has {len(pedagogy)}, need {need_p}")
    if len(general) < need_g:
        raise ValueError(f"general slot has {len(general)}, need {need_g}")

    ped = list(pedagogy)
    gen = list(general)
    random.Random(seed).shuffle(ped)
    random.Random(seed + 1).shuffle(gen)

    out: list[dict] = []
    for b in range(n_blocks):
        out.extend(ped[b * ped_per_block:(b + 1) * ped_per_block])
        out.extend(gen[b * gen_per_block:(b + 1) * gen_per_block])
    return out


def is_pedagogy(record: dict) -> bool:
    return record.get("kind") == "pedagogy"


def verify_block_layout(
    records: Sequence[dict],
    ped_per_block: int = PED_PER_BLOCK,
    gen_per_block: int = GEN_PER_BLOCK,
) -> dict:
    """PLAN §11 check 5. Raises on the first malformed block."""
    block = ped_per_block + gen_per_block
    if len(records) % block:
        raise AssertionError(
            f"{len(records)} records is not a whole number of {block}-example blocks"
        )
    n_blocks = len(records) // block
    for b in range(n_blocks):
        chunk = records[b * block:(b + 1) * block]
        for j, r in enumerate(chunk):
            want_ped = j < ped_per_block
            if is_pedagogy(r) != want_ped:
                raise AssertionError(
                    f"block {b} position {j}: expected "
                    f"{'pedagogy' if want_ped else 'general'}, got kind={r.get('kind')!r}"
                )
            has_sys = any(m["role"] == "system" for m in r["messages"])
            if has_sys != want_ped:
                raise AssertionError(
                    f"block {b} position {j}: system-message presence ({has_sys}) does not "
                    f"match stream (pedagogy={want_ped}). The Impl 2 contract is "
                    f"'system message present <=> tutor mode'."
                )
    return {
        "n_blocks": n_blocks,
        "block_size": block,
        "layout": f"{ped_per_block} pedagogy + {gen_per_block} general",
        "kinds": dict(Counter(r.get("kind") for r in records)),
        "ok": True,
    }
