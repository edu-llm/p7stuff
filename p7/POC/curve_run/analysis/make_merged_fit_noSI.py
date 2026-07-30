"""Merged (0-100 + 0-923): Math accuracy vs NO-SI new-task KL (kl_ped_noSI).
Normal aspect ratio. Fit several models, pick best by AIC/adj-R^2."""
import json, os, numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

full = json.load(open(os.path.join(ROOT, "full_0-923", "master_summary.json")))
fine = json.load(open(os.path.join(ROOT, "fine_0-100", "master_summary_0-100.json")))

def load(rows, run):
    return [{"run": run, "point": r["point"],
             "step": 0 if r["point"] == "base" else int(r["point"][1:]),
             "kl": r["kl_no"], "acc": r["acc"]} for r in rows]

pts = load(full, "full") + [p for p in load(fine, "fine") if p["point"] != "base"]
kl  = np.array([p["kl"] for p in pts]); acc = np.array([p["acc"] for p in pts])
o = np.argsort(kl); kl, acc = kl[o], acc[o]
runs = [pts[i]["run"] for i in o]; steps = np.array([pts[i]["step"] for i in o])
n = len(kl)
print(f"merged points: {n} (full={sum(r=='full' for r in runs)}, fine={sum(r=='fine' for r in runs)})")
print("\n point   run    kl_noSI    acc")
for i in range(n):
    print(f"{pts[o[i]]['point']:>5}  {runs[i]:>4}  {kl[i]:>8.4f}  {acc[i]:>6.2f}")

def stats(y, yh, k):
    ss=np.sum((y-yh)**2); tot=np.sum((y-y.mean())**2)
    r2=1-ss/tot; adj=1-(1-r2)*(n-1)/(n-k-1); aic=n*np.log(ss/n)+2*(k+1)
    return r2,adj,aic
res={}
for deg,name in [(1,"linear"),(2,"quadratic"),(3,"cubic")]:
    c=np.polyfit(kl,acc,deg); res[name]=(*stats(acc,np.polyval(c,kl),deg),c)
def expd(x,a,b,c): return c+a*np.exp(-b*x)
try:
    popt,_=curve_fit(expd,kl,acc,p0=[acc.max()-acc.min(),20,acc.min()],maxfev=20000)
    res["exp_decay"]=(*stats(acc,expd(kl,*popt),3),popt)
except Exception as e: print("exp fail",e)

print(f"\n{'model':<12}{'R2':>8}{'adjR2':>9}{'AIC':>9}   pearson(kl,acc)=%.3f"%np.corrcoef(kl,acc)[0,1])
print("-"*52)
for name,(r2,adj,aic,par) in sorted(res.items(),key=lambda kv:kv[1][2]):
    print(f"{name:<12}{r2:>8.3f}{adj:>9.3f}{aic:>9.2f}")
best=min(res,key=lambda k:res[k][2]); print(f"\nBEST by AIC: {best} (adjR2={res[best][1]:.3f})")

plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.labelsize":13,
                     "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11,"axes.linewidth":1.2})
fig,ax=plt.subplots(figsize=(8,6))
xs=np.linspace(kl.min(),kl.max(),200)
for run,mk,lab in [("full","o","full run (0-923)"),("fine","^","fine run (0-100)")]:
    idx=[i for i,r in enumerate(runs) if r==run]
    sc=ax.scatter(kl[idx],acc[idx],c=steps[idx],cmap="viridis",s=90,marker=mk,
                  zorder=3,edgecolor="k",linewidths=0.6,label=lab,vmin=0,vmax=steps.max())
def curve(name,par):
    return expd(xs,*par) if name=="exp_decay" else np.polyval(par,xs)
r2b,adjb,_,parb=res[best]
ax.plot(xs,curve(best,parb),"-",c="crimson",lw=2.3,label=f"best: {best}  $R^2$={r2b:.2f}")
r2l,_,_,parl=res["linear"]
ax.plot(xs,np.polyval(parl,xs),"--",c="gray",lw=1.8,label=f"linear  $R^2$={r2l:.2f}")
ax.set_xlabel(r"$\mathrm{KL}(\pi_0\|\pi)$;  New-task KL (NO system instruction)")
ax.set_ylabel("Math accuracy (%)")
ax.set_title("Merged: Forward KL (no SI) vs. Old Task (Math)", fontweight="bold")
ax.legend(loc="upper right",framealpha=0.92); ax.grid(alpha=0.25); ax.margins(x=0.03)
cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.02); cb.set_label("training step")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_kl_noSI_mathacc_merged.png"),dpi=150,bbox_inches="tight")
print("saved analysis/figures/fig_kl_noSI_mathacc_merged.png")
