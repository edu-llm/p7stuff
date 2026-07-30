"""0-100 step run: KL vs Old Task (Math). Wide/flat, large fonts, linear fit (line matches R^2)."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINE = os.path.join(ROOT, "fine_0-100", "master_summary_0-100.json")
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 24, "axes.labelsize": 20,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "axes.linewidth": 1.4,
})

rows = json.load(open(FINE))
pt   = [r["point"] for r in rows]
kl   = np.array([r["kl_new"] for r in rows])
acc  = np.array([r["acc"] for r in rows])
steps = [0 if p == "base" else int(p[1:]) for p in pt]

fig, ax = plt.subplots(figsize=(12, 4.3))
p = ax.scatter(kl, acc, c=steps, cmap="viridis", s=190, zorder=3, edgecolor="k", linewidths=0.7)

xs = np.linspace(kl.min(), kl.max(), 100)
m, b = np.polyfit(kl, acc, 1)
yh = m*kl + b
r2 = 1 - np.sum((acc-yh)**2)/np.sum((acc-acc.mean())**2)
ax.plot(xs, m*xs + b, "--", c="gray", lw=2.2, label=f"linear fit  $R^2$={r2:.2f}")

ax.set_xlabel(r"$\mathrm{KL}(\pi_0\|\pi)$;  New-task (Socratic) KL")
ax.set_ylabel("Math accuracy (%)")
ax.set_title("Early Training (0-100 steps): Forward KL vs. Old Task (Math)", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(alpha=0.25)
ax.margins(x=0.03)

cb = fig.colorbar(p, ax=ax, fraction=0.04, pad=0.015)
cb.set_label("training step", fontsize=16)
cb.ax.tick_params(labelsize=13)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_kl_mathacc_0-100.png"), dpi=150, bbox_inches="tight")
print("saved analysis/figures/fig_kl_mathacc_0-100.png | linear R2=%.3f | pearson=%.3f" % (r2, np.corrcoef(kl, acc)[0,1]))
