#!/usr/bin/env python3
"""Compare the empty-beam and sphere-sample (x,y) guide-alignment scans.

Produces:
  sans_maps.png     flux map with / without sample, and their ratio
  sans_iq.png       I(Q) at selected offsets, against sphere theory
  sans_summary.txt  the numbers behind both figures
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOSAMP = os.path.join(HERE, "scan_sans_nosample")
SAMP = os.path.join(HERE, "scan_sans_sample")
GRID = np.arange(-20, 21)
R_SPHERE = 100.0          # AA, as simulated


def name_of(x, y):
    return "x%+03d_y%+03d" % (x, y)


def detector_values(simfile):
    """(I, err, N) of the 'detector' component."""
    comp = None
    with open(simfile) as f:
        for line in f:
            s = line.strip()
            if s.startswith("component:"):
                comp = s.split(":", 1)[1].strip()
            elif s.startswith("values:") and comp == "detector":
                a, b, c = s.split(":", 1)[1].split()
                return float(a), float(b), float(c)
    return np.nan, np.nan, np.nan


def flux_map(root):
    I = np.full((len(GRID), len(GRID)), np.nan)
    E = np.full_like(I, np.nan)
    for iy, y in enumerate(GRID):
        for ix, x in enumerate(GRID):
            sim = os.path.join(root, name_of(x, y), "mccode.sim")
            if os.path.exists(sim):
                I[iy, ix], E[iy, ix], _ = detector_values(sim)
    return I, E


def load_iq(root, x, y):
    d = np.loadtxt(os.path.join(root, name_of(x, y), "qdet.dat"), comments="#")
    q, I, err = d[:, 0], d[:, 1], d[:, 2]
    good = (q > 0) & (I > 0)
    return q[good], I[good], err[good]


def sphere_Pq(q, R):
    """Form factor of a uniform sphere, normalised to 1 at q=0."""
    x = q * R
    f = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    return f**2


# ---------------------------------------------------------------- maps -----
In, En = flux_map(NOSAMP)
Is, Es = flux_map(SAMP)
ratio = Is / In

i0 = list(GRID).index(0)
lines = []
lines.append("EQ-SANS guide-alignment scan: empty beam vs R=%g AA spheres" % R_SPHERE)
lines.append("")
lines.append("centre (0,0):  no-sample %.4g n/s   sample %.4g n/s   ratio %.4f"
             % (In[i0, i0], Is[i0, i0], ratio[i0, i0]))
lines.append("peak flux   :  no-sample %.4g      sample %.4g"
             % (np.nanmax(In), np.nanmax(Is)))
lines.append("min  flux   :  no-sample %.4g      sample %.4g"
             % (np.nanmin(In), np.nanmin(Is)))
lines.append("scattered fraction over the whole map: mean %.4f  std %.4f  (min %.4f, max %.4f)"
             % (np.nanmean(ratio), np.nanstd(ratio), np.nanmin(ratio), np.nanmax(ratio)))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ext = [-20.5, 20.5, -20.5, 20.5]

im0 = ax[0].imshow(In, origin="lower", extent=ext, cmap="viridis")
ax[0].set_title("No sample: detector flux")
fig.colorbar(im0, ax=ax[0], label="I [n/s]")

im1 = ax[1].imshow(Is, origin="lower", extent=ext, cmap="viridis")
ax[1].set_title("With spheres: scattered flux")
fig.colorbar(im1, ax=ax[1], label="I [n/s]")

im2 = ax[2].imshow(ratio * 100, origin="lower", extent=ext, cmap="magma")
ax[2].set_title("Scattered fraction  sample/no-sample")
fig.colorbar(im2, ax=ax[2], label="[%]")

for a in ax:
    a.set_xlabel("guide x offset [mm]")
    a.set_ylabel("guide y offset [mm]")
fig.suptitle("Guide-entrance alignment scan, identical geometry with and without sample")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "sans_maps.png"), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- I(Q) -----
points = [(0, 0), (5, 0), (10, 0), (20, 0), (0, 20), (20, 20)]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) sample vs empty beam at centre
q, Iq, Eq = load_iq(SAMP, 0, 0)
qn, Iqn, _ = load_iq(NOSAMP, 0, 0)
ax[0].errorbar(q, Iq, yerr=Eq, fmt="o-", ms=3, lw=1, label="spheres")
ax[0].plot(qn, Iqn, "s--", ms=3, lw=1, color="0.5", label="empty beam")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("q [1/$\\AA$]"); ax[0].set_ylabel("I(q)")
ax[0].set_title("(0,0): sample vs empty beam")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")

# (b) I(Q) vs misalignment, absolute
for (x, y) in points:
    q, Iq, _ = load_iq(SAMP, x, y)
    ax[1].plot(q, Iq, lw=1.1, label="(%+d,%+d)" % (x, y))
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("q [1/$\\AA$]"); ax[1].set_ylabel("I(q)")
ax[1].set_title("Sample I(q) vs guide offset [mm]")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")

# (c) shape check: normalise each to its own low-q value, overlay theory
for (x, y) in points:
    q, Iq, _ = load_iq(SAMP, x, y)
    norm = Iq[(q > 0.004) & (q < 0.008)]
    if norm.size:
        ax[2].plot(q, Iq / norm.mean(), lw=1.1, label="(%+d,%+d)" % (x, y))
qt = np.logspace(np.log10(0.003), np.log10(0.15), 400)
Pt = sphere_Pq(qt, R_SPHERE)
Pt = Pt / sphere_Pq(np.array([0.006]), R_SPHERE)[0]
ax[2].plot(qt, Pt, "k--", lw=1.4, label="sphere P(q), R=%g $\\AA$" % R_SPHERE)
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_ylim(1e-4, 5)
ax[2].set_xlabel("q [1/$\\AA$]"); ax[2].set_ylabel("I(q) / I(low q)")
ax[2].set_title("Shape, normalised — does misalignment distort I(q)?")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, which="both")

fig.suptitle("EQ-SANS I(q): hard spheres R=%g $\\AA$, $\\Phi$=0.01" % R_SPHERE)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "sans_iq.png"), dpi=130)
plt.close(fig)

# ---- theory check + shape stability (grouped for adequate statistics) ----
q, Iq, _ = load_iq(SAMP, 0, 0)
lines.append("")
lines.append("q range covered: %.4f - %.4f 1/AA" % (q.min(), q.max()))
lines.append("sphere first form-factor minimum (theory): q = %.4f 1/AA" % (4.493 / R_SPHERE))

qg = np.logspace(np.log10(0.004), np.log10(0.09), 40)
nrm = np.interp(np.array([0.006]), q, Iq)[0]
sim = np.interp(qg, q, Iq) / nrm
th = sphere_Pq(qg, R_SPHERE) / sphere_Pq(np.array([0.006]), R_SPHERE)[0]
lines.append("centre I(q) vs sphere P(q): median ratio %.3f" % np.median(sim / th))

# Per-point shape tests are normalisation-noise dominated, so average points
# by |offset| radius before comparing shapes.
qs = np.logspace(np.log10(0.004), np.log10(0.10), 35)
bands = {"0-3 mm": (0, 3), "7-10 mm": (7, 10), "14-17 mm": (14, 17), "18-21 mm": (18, 21)}
curves, sizes = {}, {}
for label, (lo, hi) in bands.items():
    pts = [(x, y) for y in GRID for x in GRID if lo <= np.hypot(x, y) <= hi]
    acc = np.zeros_like(qs)
    for (x, y) in pts:
        qq, II, _ = load_iq(SAMP, x, y)
        acc += np.interp(qs, qq, II)
    curves[label] = acc / len(pts)
    sizes[label] = len(pts)

ref = curves["0-3 mm"]
lines.append("")
lines.append("I(q) shape vs guide misalignment (points averaged per band):")
for label, c in curves.items():
    sc = np.median(ref[qs < 0.012] / c[qs < 0.012])
    r = (c * sc) / ref
    lines.append("  %-9s (n=%3d)  median %.3f  std %.3f  max deviation %.1f%%"
                 % (label, sizes[label], np.median(r), np.std(r), 100 * np.max(np.abs(r - 1))))

open(os.path.join(HERE, "sans_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nsaved sans_maps.png, sans_iq.png, sans_summary.txt")
