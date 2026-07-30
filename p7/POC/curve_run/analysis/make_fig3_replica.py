"""Draft replica of RL's Razor Figure 3 (3-panel layout) using our SFT-checkpoint sweep,
MERGED across the full run (0-923) and the fine run (0-100).

Panels mirror the paper's organization:
  (Left)   Learning-Forgetting trade-off:  x = new-task performance, y = prior-task performance
  (Middle) KL predicts forgetting:          x = new-task KL,          y = forgetting (pts), linear fit
  (Right)  New-task performance vs KL:      x = new-task KL,          y = new-task performance

We only have SFT (one method) traced across training checkpoints, so each panel shows a single
SFT trajectory (the paper overlays RL + SFT + oracle). New-task "accuracy" here = pedagogy quality
(D = SFT +SI, 0-1, LLM-judge). Prior task = math/logic accuracy. KL = kl_new_SI = KL(base||sft) on
the pedagogy (new-task) distribution with the system instruction. Circles = full run, triangles = fine run.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FULL = os.path.join(ROOT, "full_0-923", "master_summary.json")
FINE = os.path.join(ROOT, "fine_0-100", "master_summary_0-100.json")
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

def load(path, run):
    return [{"run": run, "point": r["point"],
             "step": 0 if r["point"] == "base" else int(r["point"][1:]),
             "kl": r["kl_new"], "acc": r["acc"], "forget": r["forget"],
             "newp": r["pedD"] * 100} for r in json.load(open(path))]

pts = load(FULL, "full") + [p for p in load(FINE, "fine") if p["point"] != "base"]

kl     = np.array([p["kl"] for p in pts])
acc    = np.array([p["acc"] for p in pts])
forget = np.array([p["forget"] for p in pts])
newp   = np.array([p["newp"] for p in pts])
steps  = np.array([p["step"] for p in pts])
runs   = [p["run"] for p in pts]
smax   = steps.max()

MARK = {"full": ("o", "full run (0-923)"), "fine": ("^", "fine run (0-100)")}
def scatter(a, x, y):
    h = None
    for run, (mk, lab) in MARK.items():
        idx = [i for i, r in enumerate(runs) if r == run]
        h = a.scatter(x[idx], y[idx], c=steps[idx], cmap="viridis", s=95, marker=mk,
                      zorder=3, edgecolor="k", linewidths=0.5, label=lab, vmin=0, vmax=smax)
    return h

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# ---- (Left) Learning-Forgetting trade-off ----
o = np.argsort(newp)
ax[0].plot(newp[o], acc[o], "-", c="gray", alpha=0.4, zorder=1)
scatter(ax[0], newp, acc)
ax[0].set_xlabel("New-task performance  (pedagogy quality, %)")
ax[0].set_ylabel("Prior-task performance  (math accuracy, %)")
ax[0].set_title("(Left) Learning–Forgetting trade-off")
ax[0].legend(loc="lower left", fontsize=9); ax[0].grid(alpha=0.25)

# ---- (Middle) KL predicts forgetting + linear fit ----
scatter(ax[1], kl, forget)
xs = np.linspace(kl.min(), kl.max(), 100)
m, b = np.polyfit(kl, forget, 1)
yh = m*kl + b
r2 = 1 - np.sum((forget - yh)**2)/np.sum((forget - forget.mean())**2)
r_lin = np.corrcoef(kl, forget)[0, 1]
ax[1].plot(xs, m*xs + b, "--", c="gray", label=f"linear fit  $R^2$={r2:.2f}")
ax[1].set_xlabel(r"New-task KL  $\mathrm{KL}(\pi_0\|\pi)$")
ax[1].set_ylabel("Forgetting  (base − ckpt math acc, pts)")
ax[1].set_title(f"(Middle) KL predicts forgetting   (Pearson r={r_lin:.2f})")
ax[1].legend(loc="lower right"); ax[1].grid(alpha=0.25)

# ---- (Right) new-task performance vs KL ----
p2 = scatter(ax[2], kl, newp)
ax[2].set_xlabel(r"New-task KL  $\mathrm{KL}(\pi_0\|\pi)$")
ax[2].set_ylabel("New-task performance  (pedagogy quality, %)")
ax[2].set_title("(Right) New-task performance vs KL")
ax[2].grid(alpha=0.25)

cb = fig.colorbar(p2, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("training step (checkpoint)")
fig.suptitle("DRAFT — RL's Razor Fig. 3 layout, OLMo-2-1B Socratic-tutor SFT sweep (full 0-923 + fine 0-100; base anchor at step 0)",
             y=1.02, fontsize=11)
fig.savefig(os.path.join(FIG, "fig3_replica.png"), dpi=150, bbox_inches="tight")
print("saved analysis/figures/fig3_replica.png |  merged n=%d  linear R2=%.3f  Pearson r=%.3f" % (len(kl), r2, r_lin))
