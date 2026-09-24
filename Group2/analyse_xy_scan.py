#!/usr/bin/env python3
"""Build the guide-entrance (x,y) offset map from scan_guide_xy/."""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "scan_guide_xy")


def read_detector(simfile):
    """Return (I, ERR, N) of the 'detector' component from an mccode.sim."""
    comp = None
    with open(simfile) as f:
        for line in f:
            s = line.strip()
            if s.startswith("component:"):
                comp = s.split(":", 1)[1].strip()
            elif s.startswith("values:") and comp == "detector":
                a, b, c = s.split(":", 1)[1].split()
                return float(a), float(b), float(c)
    return None


xs = ys = np.arange(-20, 21)
I = np.full((len(ys), len(xs)), np.nan)
E = np.full_like(I, np.nan)

missing = 0
for iy, ymm in enumerate(ys):
    for ix, xmm in enumerate(xs):
        sim = os.path.join(SCAN, "x%+03d_y%+03d" % (xmm, ymm), "mccode.sim")
        if not os.path.exists(sim):
            missing += 1
            continue
        got = read_detector(sim)
        if got:
            I[iy, ix], E[iy, ix], _ = got

print("points read : %d / %d (missing %d)" % (np.isfinite(I).sum(), I.size, missing))

best = np.unravel_index(np.nanargmax(I), I.shape)
print("peak flux   : %.4g n/s at (x=%+d mm, y=%+d mm)" % (I[best], xs[best[1]], ys[best[0]]))
print("flux at 0,0 : %.4g n/s" % I[ys.tolist().index(0), xs.tolist().index(0)])
print("min flux    : %.4g n/s" % np.nanmin(I))

np.savetxt(os.path.join(HERE, "xy_scan_flux.txt"), I,
           header="detector flux [n/s]; rows y=-20..+20 mm, cols x=-20..+20 mm, 1 mm step")

fig = plt.figure(figsize=(14, 5))

ax0 = fig.add_subplot(1, 3, 1)
im = ax0.imshow(I, origin="lower", extent=[-20.5, 20.5, -20.5, 20.5], cmap="viridis")
ax0.plot(xs[best[1]], ys[best[0]], "r+", ms=12, mew=2, label="max")
ax0.set_xlabel("guide x offset [mm]")
ax0.set_ylabel("guide y offset [mm]")
ax0.set_title("Detector flux vs guide offset")
ax0.legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax0, label="I [n/s]")

iy0 = ys.tolist().index(0)
ix0 = xs.tolist().index(0)

ax1 = fig.add_subplot(1, 3, 2)
ax1.errorbar(xs, I[iy0, :], yerr=E[iy0, :], fmt="o-", ms=3, lw=1, color="tab:blue")
ax1.set_xlabel("guide x offset [mm]   (y = 0)")
ax1.set_ylabel("I [n/s]")
ax1.set_title("Horizontal cut")
ax1.grid(alpha=.3)

ax2 = fig.add_subplot(1, 3, 3)
ax2.errorbar(ys, I[:, ix0], yerr=E[:, ix0], fmt="o-", ms=3, lw=1, color="tab:red")
ax2.set_xlabel("guide y offset [mm]   (x = 0)")
ax2.set_ylabel("I [n/s]")
ax2.set_title("Vertical cut")
ax2.grid(alpha=.3)

fig.suptitle("EQ-SANS guide-entrance alignment scan, 41x41 @ 1 mm, N=1e6/point")
fig.tight_layout()
out = os.path.join(HERE, "xy_scan_map.png")
fig.savefig(out, dpi=130)
print("saved", out)
