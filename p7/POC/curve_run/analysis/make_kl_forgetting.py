"""KL predicts forgetting (the 'middle graph').
x = new-task (Socratic) KL, y = forgetting (base - ckpt math acc, pts). Coloring = training step.
Linear fit so the drawn line matches its reported R^2 (leverage-robust; = Pearson r^2).
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FULL = os.path.join(ROOT, "full_0-923", "master_summary.json")
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

rows = json.load(open(FULL))
pt    = [r["point"] for r in rows]
kl    = np.array([r["kl_new"] for r in rows])
forget = np.array([r["forget"] for r in rows])
steps = [0 if p == "base" else int(p[1:]) for p in pt]

fig, ax = plt.subplots(figsize=(7.2, 6.0))
p = ax.scatter(kl, forget, c=steps, cmap="viridis", s=170, zorder=3, edgecolor="k", linewidths=0.7)

xs = np.linspace(kl.min(), kl.max(), 100)
m, b = np.polyfit(kl, forget, 1)
yh = m*kl + b
r2 = 1 - np.sum((forget-yh)**2)/np.sum((forget-forget.mean())**2)
ax.plot(xs, m*xs + b, "--", c="gray", lw=2.2, label=f"linear fit  $R^2$={r2:.2f}")

ax.set_xlabel(r"New-task (Socratic) KL    $\mathrm{KL}(\pi_0\|\pi)$")
ax.set_ylabel("Forgetting (base - ckpt math acc, pts)")
ax.set_title("KL predicts forgetting")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(alpha=0.25)

cb = fig.colorbar(p, ax=ax)
cb.set_label("training step")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_kl_forgetting.png"), dpi=150, bbox_inches="tight")
print("saved analysis/figures/fig_kl_forgetting.png | linear R2=%.3f" % r2)
