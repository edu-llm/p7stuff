#!/usr/bin/env python
"""Pedagogy-NLL trajectories: Impl 5's D4 against impl4's A1 (= D0) and A3.

Two panels, and the second is the one that carries the result.

**Left — absolute ``ped_nll`` vs step.** The honest view, and on this axis the arms are
almost on top of each other: the whole spread at step 923 is a few thousandths against a
0.55-nat drop from base. Anyone shown only this panel would reasonably conclude "no
difference", which is a fair reading of the absolute scale but throws away a paired contrast.

**Right — Δ vs D0, same dialogues, same step.** D4 and A1 see the same problems in the same
block positions (the distilled pool is written in the gold pool's row order and the mix
substitution is positional), so the per-step difference is paired and a few thousandths is
readable. Zero is D0.

Why no dual axis: the two panels share the same measure in different framings, so they are
two charts rather than two y-scales on one.

The base model's ``ped_nll`` is drawn as a reference rule, not a series — it is a constant,
and giving it a line in the legend would imply it has a trajectory.

    python plot_ped_nll.py --rows ped_nll.jsonl --rows ped_nll_impl5.jsonl --out fig.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# Categorical hues, assigned in fixed order and never cycled. Validated for CVD: the
# green/orange pair sits at ΔE 8.0 under protanopia, which is the floor band and is legal
# only with secondary encoding — hence the distinct marker shapes and the direct end labels.
STYLE = {
    "impl4-A1": ("#2a78d6", "o", "D0 · Impl 2 gold\n(impl4-A1)"),
    "impl4-A3": ("#eb6834", "s", "Impl 4 · σ=1\nself-distilled replay"),
    "impl5-D4": ("#2e9e6b", "^", "Impl 5 · δ=1\nself-distilled pedagogy"),
}
ORDER = ("impl4-A1", "impl4-A3", "impl5-D4")
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e1"
SURFACE = "#fcfcfb"

#: Impl 3's `impl2-rerun` final checkpoint, from James's handoff table. A1 reproduces it to
#: within 0.002, which is what licenses putting Impl 5 on the same axis at all.
IMPL3_SFT_923 = 0.862


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", action="append", required=True,
                   help="ped_nll JSONL file(s). Repeatable.")
    p.add_argument("--out", default="fig_ped_nll.png")
    p.add_argument("--baseline", default="impl4-A1")
    p.add_argument("--title", default="Pedagogy NLL on 128 held-out gold dialogues")
    return p.parse_args()


def load(paths):
    runs, base = {}, None
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ped_nll") is None:
                continue
            if r["run"] == "base":
                base = r["ped_nll"]
            else:
                runs.setdefault(r["run"], {})[int(r["step"])] = float(r["ped_nll"])
    return runs, base


def main():
    args = parse_args()
    runs, base = load(args.rows)
    present = [n for n in ORDER if n in runs] + [n for n in sorted(runs) if n not in ORDER]
    if not present:
        raise SystemExit(f"no ped_nll rows found in {args.rows}")
    print(f"runs: {', '.join(f'{n} ({len(runs[n])} pts)' for n in present)}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4), facecolor=SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)
        a.set_xscale("log")
        a.set_xlabel("optimizer step (log)", color=MUTED, fontsize=10)

    # -- left: absolute --------------------------------------------------------
    if base is not None:
        ax.axhline(base, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"base π₀  {base:.3f}", xy=(1.05, base), xytext=(0, 5),
                    textcoords="offset points", color=MUTED, fontsize=9)
    for name in present:
        c, mk, lab = STYLE.get(name, ("#7a7a7a", "d", name))
        xs = sorted(runs[name])
        ys = [runs[name][x] for x in xs]
        ax.plot(xs, ys, color=c, lw=2, marker=mk, ms=4.5, mew=0, zorder=3,
                label=lab.replace("\n", " "))
    ax.axhline(IMPL3_SFT_923, color=MUTED, lw=1, ls=":", zorder=1)
    ax.annotate(f"Impl 3 SFT @923  {IMPL3_SFT_923:.3f}", xy=(1.05, IMPL3_SFT_923),
                xytext=(0, -13), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_ylabel("ped_nll  (lower = fits gold tutoring better)", color=MUTED, fontsize=10)
    ax.set_title("Absolute — the arms are nearly indistinguishable here",
                 color=INK, fontsize=11, loc="left", pad=10)
    # Lower left: the curves fall left-to-right, and upper-right collides with the base rule.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

    # -- right: paired difference ---------------------------------------------
    ref = runs.get(args.baseline)
    if ref:
        bx.axhline(0, color=MUTED, lw=1.2, zorder=1)
        bx.annotate(f"{args.baseline} = 0", xy=(1.05, 0), xytext=(0, 5),
                    textcoords="offset points", color=MUTED, fontsize=9)
        for name in present:
            if name == args.baseline:
                continue
            c, mk, lab = STYLE.get(name, ("#7a7a7a", "d", name))
            xs = sorted(x for x in runs[name] if x in ref)
            ys = [1000 * (runs[name][x] - ref[x]) for x in xs]
            bx.plot(xs, ys, color=c, lw=2, marker=mk, ms=4.5, mew=0, zorder=3)
            # Direct label at the line end — the secondary encoding the CVD floor requires.
            bx.annotate(lab, xy=(xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        color=c, fontsize=9, va="center")
        bx.set_ylabel(f"ped_nll − {args.baseline}   (millinats)", color=MUTED, fontsize=10)
        bx.set_title("Paired vs D0 — same dialogues, same block positions",
                     color=INK, fontsize=11, loc="left", pad=10)
        bx.margins(x=0.28)

    fig.suptitle(args.title, color=INK, fontsize=13, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.925,
             "Held-out dialogues are gold and were never distilled, so D4 is scored on a "
             "target distribution it was not trained on — some gap is expected by "
             "construction.", color=MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")

    print("\nfinal step:")
    for name in present:
        last = max(runs[name])
        d = ""
        if ref and last in ref and name != args.baseline:
            d = f"  ({1000 * (runs[name][last] - ref[last]):+.1f} millinats vs {args.baseline})"
        print(f"  {name:12s} step {last:>4}  ped_nll {runs[name][last]:.4f}{d}")


if __name__ == "__main__":
    main()
