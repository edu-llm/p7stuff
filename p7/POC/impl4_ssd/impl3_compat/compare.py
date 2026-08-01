#!/usr/bin/env python
"""Merge Impl 4's per-checkpoint rows with Impl 3's, gate on A1, and plot both.

Three jobs:

1. **Protocol check.** Every row on both sides must carry the same ``protocol`` stamp. Impl 3
   corrupted a results file twice by appending rows measured under different rules, which is
   why the stamp exists; merging across stamps is refused rather than warned about.

2. **The A1 gate.** Impl 4's ``A1`` arm is vanilla Impl 2 on the same data as Impl 3's
   ``impl2-rerun``, so its final checkpoint should reproduce their numbers. That single
   comparison validates the canonical SI, the 250-item set, the KL truncation rule, the
   assistant-only masking behind pedagogy NLL, both compat shims, the pinned dataset revision,
   and the training config — all at once. Deltas are reported as a share of the axis range
   *measured from their own 194 rows*, so "close" means close relative to the spread the
   figure actually shows.

3. **The merged figure.** Their ``eval/plot_figure3.py`` is the canonical figure, but it styles
   every ``variant is None`` run as a black X — so merging eight Impl 4 arms into it yields nine
   indistinguishable black curves. This draws a two-panel version that separates them, and the
   merged JSONL it writes is what to hand back for their script once they add a marker entry.

Usage:
    python impl3_compat/compare.py
    python impl3_compat/compare.py --impl4 work/out/ckpt_sweep_impl4.jsonl --no_plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Impl 3's impl2-rerun @ step 923 (BUNDLE_README.md §3) — what A1 should reproduce.
GATE = {
    "kl_new_SI": 0.7607,
    "kl_ped_noSI": 0.1500,
    "ped_nll": 0.862,
    "math_hint": 0.212,
    "math_bare": 0.456,
    "math_hint_commit": 0.904,
    "math_hint_deflect": 0.476,
}
GATE_RUN = "impl2-rerun"
SUSPICIOUS_FRAC = 0.05      # |delta| above this share of the axis range gets flagged


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl3", default=None,
                   help="Their results jsonl (default: <bundle>/out/ckpt_sweep_bare_hint250.jsonl).")
    p.add_argument("--impl4", default=str(HERE / "work/out/ckpt_sweep_impl4.jsonl"))
    p.add_argument("--out", default=str(HERE / "work/out/merged.jsonl"))
    p.add_argument("--fig", default=str(HERE / "work/out/impl3_vs_impl4.png"))
    p.add_argument("--gate_arm", default="impl4-A1",
                   help="Which of our runs is the vanilla-Impl-2 reference.")
    p.add_argument("--exclude", default="poc-c923",
                   help="Comma-separated runs to leave out of the figure (their default).")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def find_impl3_results(explicit) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    setup = HERE / "work/compat_setup.json"
    if setup.exists():
        bundle = Path(json.loads(setup.read_text())["bundle"])
        cand = bundle / "out/ckpt_sweep_bare_hint250.jsonl"
        if cand.exists():
            return cand
    raise SystemExit("cannot locate their results jsonl; pass --impl3 PATH")


def axis_ranges(rows: list[dict]) -> dict:
    """Observed span of each metric across their rows — the scale 'close' is judged against."""
    out = {}
    for k in GATE:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = max(vals) - min(vals)
    return out


def gate_table(theirs: list[dict], ours: list[dict], gate_arm: str) -> bool:
    their_final = next((r for r in theirs
                        if r["run"] == GATE_RUN and r["step"] == max(
                            x["step"] for x in theirs if x["run"] == GATE_RUN)), None)
    ours_arm = [r for r in ours if r["run"] == gate_arm]
    print("=" * 78)
    print(f"A1 GATE — {gate_arm} @ final vs their {GATE_RUN} @ 923")
    print("=" * 78)
    if not ours_arm:
        print(f"  {gate_arm} has no rows yet — train and eval it first. Nothing to gate.")
        return False
    mine = max(ours_arm, key=lambda r: r["step"])
    ranges = axis_ranges(theirs)
    print(f"  our step {mine['step']}   |   axis range from their {len(theirs)} rows\n")
    print(f"  {'metric':<20} {'theirs':>9} {'ours':>9} {'delta':>9} {'% of range':>11}  flag")
    worst = 0.0
    for k, want in GATE.items():
        got = mine.get(k)
        if got is None:
            print(f"  {k:<20} {want:>9.4f} {'--':>9} {'--':>9} {'--':>11}  MISSING")
            continue
        d = got - want
        rng = ranges.get(k) or 0.0
        frac = abs(d) / rng if rng else 0.0
        worst = max(worst, frac)
        flag = "ok" if frac <= SUSPICIOUS_FRAC else "CHECK"
        print(f"  {k:<20} {want:>9.4f} {got:>9.4f} {d:>+9.4f} {frac:>10.1%}  {flag}")
    print(f"\n  worst deviation: {worst:.1%} of axis range "
          f"(flagging above {SUSPICIOUS_FRAC:.0%})")
    if worst > SUSPICIOUS_FRAC:
        print("  Localize before trusting the other arms:\n"
              "    ped_nll off      -> the common/chat.py masking shim, or the dataset revision\n"
              "    kl_* off         -> canonical_si.txt, or the KL contexts\n"
              "    math_* off       -> the item set, or generation settings\n"
              "    all slightly off -> ordering (our 24/8 blocks vs their shuffle) or dtype")
    else:
        print("  A1 reproduces their baseline: the SI, item set, masking, shims, dataset "
              "revision and training config all agree.")
    return worst <= SUSPICIOUS_FRAC


def plot(theirs, ours, fig_path, exclude):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping the figure (pass --no_plot to silence).")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    base = next((r for r in theirs if r["run"] == "base"), None)

    def series(rows):
        by = defaultdict(list)
        for r in rows:
            if r["run"] == "base" or r["run"] in exclude or r.get("kl_ped_noSI") is None:
                continue
            by[r["run"]].append(r)
        for v in by.values():
            v.sort(key=lambda r: r["step"])
        return by

    theirs_by, ours_by = series(theirs), series(ours)
    temps = sorted({r[0].get("temperature") for r in theirs_by.values() if r[0].get("temperature")})
    trank = {t: (i / max(1, len(temps) - 1)) for i, t in enumerate(temps)}
    cmap = plt.get_cmap("turbo")
    ours_cmap = plt.get_cmap("tab10")

    for xk, yk, ax, title in (("kl_ped_noSI", "math_hint", axes[0],
                               "KL (no SI) vs GSM8K retention (hinted)"),
                              ("ped_nll", "math_hint", axes[1],
                               "New-task NLL vs GSM8K retention (hinted)")):
        for name, rows in sorted(theirs_by.items()):
            v, T = rows[0].get("variant"), rows[0].get("temperature")
            if v in ("a", "b") and T:
                c, m, lbl = cmap(0.06 + 0.88 * trank[T]), {"a": "s", "b": "o"}[v], name.replace("impl3-", "")
            else:
                c, m, lbl = "black", "X", ("SFT" if name == GATE_RUN else name)
            ax.plot([r[xk] for r in rows], [r[yk] for r in rows], "-", color=c, alpha=0.75,
                    lw=1.6 if m == "X" else 1.1, marker=m, ms=5, mec="k", mew=0.3, label=lbl)
        for i, (name, rows) in enumerate(sorted(ours_by.items())):
            ax.plot([r[xk] for r in rows], [r[yk] for r in rows], "--",
                    color=ours_cmap(i % 10), marker="^", ms=7, mec="k", mew=0.4,
                    lw=1.8, label=name)
        if base:
            ax.scatter([base[xk]], [base[yk]], marker="*", s=340, color="black", zorder=7)
            ax.annotate("base", (base[xk], base[yk]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8)
        ax.set_xlabel({"kl_ped_noSI": r"New-task KL, no SI   $\mathrm{KL}(\pi_0\|\pi)$",
                       "ped_nll": "New-task performance: held-out pedagogy NLL"}[xk])
        ax.set_ylabel("Prior-task score: GSM8K accuracy (hinted)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[1].invert_xaxis()      # lower NLL = learned more; keep "better" to the right
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    fig.suptitle("Impl 3 (solid, squares=a / circles=b) vs Impl 4 (dashed triangles)", fontsize=11)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")


def main():
    args = parse_args()
    their_path = find_impl3_results(args.impl3)
    theirs, ours = load(their_path), load(Path(args.impl4))
    exclude = {s for s in args.exclude.split(",") if s}

    protocols = {r.get("protocol") for r in theirs} | {r.get("protocol") for r in ours}
    print(f"theirs: {len(theirs)} rows from {their_path.name}")
    print(f"ours  : {len(ours)} rows from {Path(args.impl4).name}")
    if len(protocols) != 1:
        raise SystemExit(
            "REFUSING to merge — rows were measured under different protocols:\n  "
            + "\n  ".join(sorted(str(p) for p in protocols))
            + "\nThe stamp covers the KL context rule, the math conditions and a hash of the item\n"
              "ids, so these columns do not mean the same thing. Re-run the odd side rather than\n"
              "mixing them.")
    print(f"protocol (shared): {protocols.pop()}\n")

    ok = gate_table(theirs, ours, args.gate_arm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in theirs + ours:
            f.write(json.dumps(r) + "\n")
    print(f"\nmerged {len(theirs) + len(ours)} rows -> {out}")
    print("  Their plot_figure3.py styles every variant=null run as a black X, so ours collide "
          "there.\n  Ask them to add:  MARKER = {\"a\": \"s\", \"b\": \"o\", \"impl4\": \"^\"}")

    if not args.no_plot:
        plot(theirs, ours, args.fig, exclude)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
