#!/usr/bin/env python
"""Training-loss curves, with the caveat that makes them readable.

**These curves are not comparable in level across arms, and the gap is not a result.** Each
arm's loss is measured against *its own* targets: A1's against gold SocraTeach, D4's against
π₀'s paraphrases of it. A model predicting its own distribution's paraphrases has an easier
job than one predicting a human writer's phrasing, so D4's loss sits lower **by construction**
— that is the SDFT premise restated as a training curve, not evidence about pedagogy.

What the curves *are* good for: shape. Divergence, a stalled schedule, a loss spike, or a
warmup that behaves differently under rewritten targets all show up here and nowhere else.

    python plot_loss.py --log A1=path/train.log --log D4=path/train.log --out loss.png
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

STYLE = {"A1": ("#2a78d6", "D0 · Impl 2 gold (impl4-A1)"),
         "A3": ("#eb6834", "Impl 4 · σ=1"),
         "D4": ("#2e9e6b", "Impl 5 · δ=1")}
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#6b6b6b", "#e4e4e1", "#fcfcfb"

#: HF logs one dict per logging_steps; `epoch` scales to the step count.
_LOG = re.compile(r"\{'loss':.*?\}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--steps", type=int, default=923)
    p.add_argument("--out", default="fig_loss.png")
    return p.parse_args()


def series(path: str, total: int):
    xs, ys = [], []
    for m in _LOG.findall(Path(path).read_text(encoding="utf-8", errors="replace")):
        try:
            d = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        if "loss" in d and "epoch" in d:
            xs.append(float(d["epoch"]) * total)
            ys.append(float(d["loss"]))
    return xs, ys


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    for spec in args.log:
        name, _, path = spec.partition("=")
        xs, ys = series(path, args.steps)
        if not xs:
            print(f"  no loss lines in {path}")
            continue
        c, lab = STYLE.get(name, ("#7a7a7a", name))
        ax.plot(xs, ys, color=c, lw=1.8, zorder=3, label=lab)
        print(f"  {name}: {len(xs)} points, final {ys[-1]:.4f}")

    ax.set_xlabel("optimizer step", color=MUTED, fontsize=10)
    ax.set_ylabel("training loss (each arm on its OWN targets)", color=MUTED, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Training loss — shape is comparable, level is not",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.925,
             "D4's targets are π₀'s own paraphrases, so its loss is lower by construction. "
             "The gap is the premise, not a result.",
             color=MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
