#!/usr/bin/env python3
"""Plots for the cylinder training set.

  cylinder_dataset.png      I(q) variation, parameter coverage, noise vs q
  cylinder_regression.png   predicted vs true for each target
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "cylinder_trainingset.npz"), allow_pickle=True)
q, X, Xe, y = d["q"], d["X"], d["X_err"], d["y"]
names = [str(s) for s in d["y_names"]]
I = 10.0**X                      # back to linear intensity
R = y[:, names.index("radius")]

# ----------------------------------------------------------------- dataset --
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

order = np.argsort(R)
sel = order[:: max(1, len(order) // 120)]
cmap = plt.get_cmap("viridis")
rn = (np.log10(R[sel]) - np.log10(R.min())) / (np.log10(R.max()) - np.log10(R.min()))
for k, i in enumerate(sel):
    ax[0].plot(q, I[i], lw=0.7, alpha=0.7, color=cmap(rn[k]))
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("q [1/$\\AA$]"); ax[0].set_ylabel("I(q)")
ax[0].set_title("I(q) across the sweep (colour = radius)")
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=matplotlib.colors.LogNorm(R.min(), R.max()))
fig.colorbar(sm, ax=ax[0], label="radius [$\\AA$]")
ax[0].grid(alpha=.3, which="both")

sc = ax[1].scatter(y[:, names.index("radius")], y[:, names.index("length")],
                   c=y[:, names.index("sld")], s=7, cmap="plasma")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("radius [$\\AA$]"); ax[1].set_ylabel("length [$\\AA$]")
ax[1].set_title("Latin-hypercube coverage (n=%d)" % len(y))
fig.colorbar(sc, ax=ax[1], label="sld [1e-6/$\\AA^2$]")
ax[1].grid(alpha=.3, which="both")

ax[2].plot(q, 100 * np.median(Xe, axis=0), "o-", ms=3, color="tab:red",
           label="median over samples")
ax[2].fill_between(q, 100 * np.percentile(Xe, 16, axis=0),
                   100 * np.percentile(Xe, 84, axis=0), alpha=.25, color="tab:red")
ax[2].axhline(24, ls="--", color="k", lw=1,
              label="instrument $\\Delta q/q \\approx 24\\%$")
ax[2].set_xscale("log")
ax[2].set_xlabel("q [1/$\\AA$]"); ax[2].set_ylabel("relative error [%]")
ax[2].set_title("Monte-Carlo noise per q-bin")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")

fig.suptitle("EQ-SANS cylinder training set: %d samples, ncount=5e6" % len(y))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cylinder_dataset.png"), dpi=130)
plt.close(fig)

# -------------------------------------------------------------- regression --
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2))
summary = []
for j, nm in enumerate(names):
    m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr[:, j])
    p = m.predict(Xte)
    r2 = r2_score(yte[:, j], p)
    summary.append((nm, r2))
    a = ax[j]
    a.scatter(yte[:, j], p, s=8, alpha=.5)
    lo, hi = yte[:, j].min(), yte[:, j].max()
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_xlabel("true " + nm); a.set_ylabel("predicted " + nm)
    verdict = "good" if r2 > 0.9 else "weak" if r2 > 0.3 else "not learnable"
    a.set_title("%s\n$R^2$=%.3f  (%s)" % (nm, r2, verdict))
    a.grid(alpha=.3)
fig.suptitle("Baseline random forest on log$_{10}$ I(q), train %d / test %d"
             % (len(Xtr), len(Xte)))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cylinder_regression.png"), dpi=130)
plt.close(fig)

print("saved cylinder_dataset.png, cylinder_regression.png")
for nm, r2 in summary:
    print("  %-12s R2=%.3f" % (nm, r2))
