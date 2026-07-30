"""Merge the 0-100 fine run with the original 0-900 run; fit several models to
Math accuracy vs new-task KL and pick the best by adjusted R^2 (+AIC)."""
import json, os, numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

full = json.load(open(os.path.join(ROOT, "full_0-923", "master_summary.json")))       # base + c100..c923
fine = json.load(open(os.path.join(ROOT, "fine_0-100", "master_summary_0-100.json"))) # base + c20..c100

def load(rows, run):
    out = []
    for r in rows:
        out.append({"run": run, "point": r["point"],
                    "step": 0 if r["point"] == "base" else int(r["point"][1:]),
                    "kl": r["kl_new"], "acc": r["acc"]})
    return out

pts = load(full, "full")
# drop the duplicate base from the fine run (identical 0,20); keep everything else incl. its c100
pts += [p for p in load(fine, "fine") if p["point"] != "base"]

kl  = np.array([p["kl"] for p in pts])
acc = np.array([p["acc"] for p in pts])
order = np.argsort(kl)
kl, acc = kl[order], acc[order]
runs  = [pts[i]["run"] for i in order]
steps = np.array([pts[i]["step"] for i in order])
n = len(kl)
print(f"merged points: {n}  (full={sum(r=='full' for r in runs)}, fine={sum(r=='fine' for r in runs)})")

def stats(y, yh, k):
    ss_res = np.sum((y-yh)**2); ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    adj = 1 - (1-r2)*(n-1)/(n-k-1)          # k = number of predictors
    aic = n*np.log(ss_res/n) + 2*(k+1)
    return r2, adj, aic

results = {}

# polynomials
for deg, name in [(1,"linear"), (2,"quadratic"), (3,"cubic")]:
    c = np.polyfit(kl, acc, deg)
    yh = np.polyval(c, kl)
    results[name] = (*stats(acc, yh, deg), c)

# exponential decay to an asymptote: acc = c + a*exp(-b*kl)
def expdecay(x, a, b, c): return c + a*np.exp(-b*x)
try:
    p0 = [acc.max()-acc.min(), 3.0, acc.min()]
    popt, _ = curve_fit(expdecay, kl, acc, p0=p0, maxfev=20000)
    yh = expdecay(kl, *popt)
    results["exp_decay"] = (*stats(acc, yh, 3), popt)
except Exception as e:
    print("exp_decay failed:", e)

# square-root / sublinear: acc = a*sqrt(kl) + b*kl + c  (captures fast early drop)
def sqrtmod(x, a, b, c): return a*np.sqrt(x) + b*x + c
try:
    popt, _ = curve_fit(sqrtmod, kl, acc, maxfev=20000)
    yh = sqrtmod(kl, *popt)
    results["sqrt+lin"] = (*stats(acc, yh, 3), popt)
except Exception as e:
    print("sqrt+lin failed:", e)

print(f"\n{'model':<12}{'R2':>8}{'adjR2':>9}{'AIC':>9}   params")
print("-"*60)
for name,(r2,adj,aic,par) in sorted(results.items(), key=lambda kv: kv[1][2]):
    pr = np.array2string(np.array(par), precision=3, separator=",")
    print(f"{name:<12}{r2:>8.3f}{adj:>9.3f}{aic:>9.2f}   {pr}")

best = min(results, key=lambda k: results[k][2])   # lowest AIC
print(f"\nBEST by AIC: {best}  (adjR2={results[best][1]:.3f})")

# ---- plot merged scatter + best fit (and linear for reference) ----
plt.rcParams.update({"font.size":18,"axes.titlesize":23,"axes.labelsize":20,
                     "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":15,"axes.linewidth":1.4})
fig, ax = plt.subplots(figsize=(12,5))
xs = np.linspace(kl.min(), kl.max(), 200)
for run, mk, lab in [("full","o","full run (0-923)"), ("fine","^","fine run (0-100)")]:
    idx = [i for i,r in enumerate(runs) if r==run]
    sc = ax.scatter(kl[idx], acc[idx], c=steps[idx], cmap="viridis", s=170, marker=mk,
                    zorder=3, edgecolor="k", linewidths=0.7, label=lab, vmin=0, vmax=steps.max())

def curve(name, par):
    if name in ("linear","quadratic","cubic"): return np.polyval(par, xs)
    if name=="exp_decay": return expdecay(xs, *par)
    if name=="sqrt+lin":  return sqrtmod(xs, *par)

r2b, adjb, aicb, parb = results[best]
ax.plot(xs, curve(best, parb), "-", c="crimson", lw=2.6, zorder=2,
        label=f"best: {best}  $R^2$={r2b:.2f} (adj {adjb:.2f})")
r2l, adjl, aicl, parl = results["linear"]
ax.plot(xs, np.polyval(parl, xs), "--", c="gray", lw=2.0, zorder=1,
        label=f"linear  $R^2$={r2l:.2f}")

ax.set_xlabel(r"$\mathrm{KL}(\pi_0\|\pi)$;  New-task (Socratic) KL")
ax.set_ylabel("Math accuracy (%)")
ax.set_title("Merged curve (0-100 + 0-923): Forward KL vs. Old Task (Math)", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.92, fontsize=13)
ax.grid(alpha=0.25); ax.margins(x=0.02)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.015); cb.set_label("training step", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_kl_mathacc_merged.png"), dpi=150, bbox_inches="tight")
print("saved analysis/figures/fig_kl_mathacc_merged.png")
