"""Small, dependency-light statistics for the MRBench eval.

Ported from the trustworthy bits of ``colab_eval.ipynb`` so the ``.py`` pipeline
reports the same error bars, not just point estimates:

  - :func:`bootstrap_ci`   — mean + 95% percentile bootstrap CI over dialogues.
  - :func:`paired_delta`   — paired (same-dialogue) effect of one condition vs
                             another, with a "reliable" flag when the CI excludes 0.
  - :func:`cohens_kappa`   — chance-corrected agreement (judge vs human labels).

Uses ``numpy`` if available (fast, vectorised) and otherwise falls back to the
standard library, so this imports cleanly in the CPU-only scoring environment
declared in ``requirements.txt``.

Bootstrap resampling needs a seeded RNG for reproducibility; pass ``seed``.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

try:  # optional fast path
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only in stdlib-only envs
    _np = None


Number = float


def _clean(values: Sequence[float | None]) -> list[float]:
    """Drop ``None``/NaN entries (invalid judge outputs) before aggregating."""
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out.append(f)
    return out


def bootstrap_ci(
    values: Sequence[float | None],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Return (mean, lo, hi) with a percentile bootstrap CI, ignoring None/NaN.

    Returns (None, None, None) if there are no valid values.
    """
    x = _clean(values)
    if not x:
        return (None, None, None)
    lo_pct = (100.0 - ci) / 2.0
    hi_pct = 100.0 - lo_pct

    if _np is not None:
        arr = _np.asarray(x, dtype=float)
        rng = _np.random.default_rng(seed)
        boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
        return (float(arr.mean()), float(_np.percentile(boot, lo_pct)),
                float(_np.percentile(boot, hi_pct)))

    rng = random.Random(seed)
    n = len(x)
    means = []
    for _ in range(n_boot):
        sample = [x[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(x) / n
    return (mean, _percentile(means, lo_pct), _percentile(means, hi_pct))


def paired_delta(
    baseline: Sequence[float | None],
    treatment: Sequence[float | None],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> dict[str, float | bool | None]:
    """Paired effect (treatment - baseline) on the SAME items.

    ``baseline`` and ``treatment`` must be aligned element-for-element (same
    dialogue at the same index). Pairs where either side is None/NaN are dropped.
    Returns {mean, lo, hi, n, reliable} where ``reliable`` is True iff the 95%
    CI excludes 0.
    """
    if len(baseline) != len(treatment):
        raise ValueError("baseline and treatment must be the same length (paired).")
    diffs: list[float] = []
    for b, t in zip(baseline, treatment):
        bt = _clean([b])
        tt = _clean([t])
        if bt and tt:
            diffs.append(tt[0] - bt[0])
    if not diffs:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "reliable": False}
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)
    reliable = bool(lo is not None and hi is not None and (lo > 0 or hi < 0))
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(diffs), "reliable": reliable}


def cohens_kappa(human: Sequence[str], machine: Sequence[str], labels: Sequence[str]) -> float:
    """Cohen's kappa between two label sequences over a fixed label set.

    <0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 strong.
    """
    if len(human) != len(machine):
        raise ValueError("human and machine label sequences must be the same length.")
    n = len(human)
    if n == 0:
        return 0.0
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = [[0.0] * k for _ in range(k)]
    for h, m in zip(human, machine):
        if h in idx and m in idx:
            obs[idx[h]][idx[m]] += 1.0
    total = sum(sum(row) for row in obs)
    if total == 0:
        return 0.0
    po = sum(obs[i][i] for i in range(k)) / total
    row_marg = [sum(obs[i]) / total for i in range(k)]
    col_marg = [sum(obs[r][c] for r in range(k)) / total for c in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k))
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted list (numpy-free)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def fmt_ci(mean: float | None, lo: float | None, hi: float | None) -> str:
    """Pretty '0.812 [0.780, 0.844]' formatting (em dash when empty)."""
    if mean is None:
        return "—"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


if __name__ == "__main__":
    # Tiny self-check (no network, no GPU).
    xs = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0]
    print("bootstrap_ci:", fmt_ci(*bootstrap_ci(xs, n_boot=500, seed=0)))
    base = [0.0, 0.5, 0.5, 1.0]
    treat = [0.5, 1.0, 1.0, 1.0]
    print("paired_delta:", paired_delta(base, treat, n_boot=500, seed=0))
    h = ["Yes", "No", "Yes", "To some extent", "No"]
    m = ["Yes", "No", "To some extent", "To some extent", "No"]
    print("cohens_kappa:", round(cohens_kappa(h, m, ["Yes", "To some extent", "No"]), 3))
