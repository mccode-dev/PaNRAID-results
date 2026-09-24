#!/usr/bin/env python3
"""Detector image per (x,y) scan point, plus a subsampled montage.

  python3 plot_detector_images.py            # per-point PNGs + montage
  python3 plot_detector_images.py montage    # montage only (PNGs already made)

Each per-point figure is a joint plot: the 2D detector image, with the
x-projection (summed over y) below it and the y-projection (summed over x)
to its right.

All images and all projections share one scale across the whole scan, taken
from the (0,0) point (the flux maximum), so panels are directly comparable.
"""
import os, sys, glob
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "scan_guide_xy")
NY = NX = 256
EXTENT = [-50, 50, -70, 70]          # detector is 1.0 x 1.4 m, in cm
XG = np.linspace(EXTENT[0], EXTENT[1], NX)
YG = np.linspace(EXTENT[2], EXTENT[3], NY)
MONT_STEP = 4                        # montage samples every 4 mm
MONT_CROP = (-30, 30, -45, 45)       # montage zooms in; beam only spans ~+-20 cm


def load_image(folder):
    """First block of a McStas 2D .dat is the intensity array."""
    raw = np.loadtxt(os.path.join(folder, "detector.dat"), comments="#")
    return raw[:NY, :]


def load_spectrum(folder):
    """L_monitor output: columns are L, I, I_err, N."""
    d = np.loadtxt(os.path.join(folder, "lambda_det.dat"), comments="#")
    return d[:, 0], d[:, 1]


def name_of(xmm, ymm):
    return "x%+03d_y%+03d" % (xmm, ymm)


def global_scales():
    """Shared image and projection limits, from the brightest (0,0) point.

    Scaling the image to the single hottest pixel leaves every panel nearly
    black, so clip at the 99.5th percentile instead.
    """
    ref = os.path.join(SCAN, name_of(0, 0))
    img = load_image(ref)
    refL, refI = load_spectrum(ref)
    return dict(vmax=float(np.percentile(img, 99.5)),
                px=float(img.sum(axis=0).max()),
                py=float(img.sum(axis=1).max()),
                refL=refL, refI=refI, lmax=float(refI.max()))


def plot_one(args):
    folder, sc = args
    img = load_image(folder)
    tag = os.path.basename(folder)
    prof_x = img.sum(axis=0)
    prof_y = img.sum(axis=1)
    tot = img.sum()
    cx = (prof_x * XG).sum() / tot
    cy = (prof_y * YG).sum() / tot
    # plot projections in 1e6 n/s so matplotlib doesn't add an offset label
    # that collides with the axis title on the narrow side panel
    prof_x = prof_x / 1e6
    prof_y = prof_y / 1e6

    lam, spec = load_spectrum(folder)

    fig = plt.figure(figsize=(5.8, 7.9))
    # outer split keeps the joint plot tight while giving the spectrum its own room
    outer = fig.add_gridspec(2, 1, height_ratios=(4.6, 1.5), hspace=0.30)
    gs = outer[0].subgridspec(2, 3, width_ratios=(4, 1.3, 0.22),
                              height_ratios=(4, 1.3),
                              wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[0, 2])
    axl = fig.add_subplot(outer[1])

    im = ax.imshow(img, origin="lower", extent=EXTENT, cmap="viridis",
                   vmin=0, vmax=sc["vmax"], aspect="auto")
    ax.axvline(cx, color="w", lw=0.6, ls="--", alpha=0.7)
    ax.axhline(cy, color="w", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Y position [cm]")
    ax.tick_params(labelbottom=False)
    ax.set_title("%s    I = %.3g n/s\ncentroid (%.2f, %.2f) cm" % (tag, tot, cx, cy),
                 fontsize=10)

    fig.colorbar(im, cax=cax, label="I [n/s/pixel]")

    # x-projection (summed over y), below the image
    axb.plot(XG, prof_x, lw=1.0, color="tab:blue")
    axb.fill_between(XG, prof_x, color="tab:blue", alpha=0.25)
    axb.axvline(cx, color="k", lw=0.6, ls="--", alpha=0.6)
    axb.set_xlabel("X position [cm]")
    axb.set_ylabel("sum over y\n[$10^6$ n/s]", fontsize=8)
    axb.set_ylim(0, sc["px"] / 1e6 * 1.05)
    axb.tick_params(labelsize=8)
    axb.grid(alpha=.3)

    # y-projection (summed over x), right of the image
    axr.plot(prof_y, YG, lw=1.0, color="tab:red")
    axr.fill_betweenx(YG, prof_y, color="tab:red", alpha=0.25)
    axr.axhline(cy, color="k", lw=0.6, ls="--", alpha=0.6)
    axr.set_xlabel("sum over x\n[$10^6$ n/s]", fontsize=8)
    axr.set_xlim(0, sc["py"] / 1e6 * 1.05)
    axr.tick_params(labelleft=False, labelsize=8)
    axr.grid(alpha=.3)

    # wavelength spectrum at the detector, with the (0,0) point as reference
    mlam = (sc["refI"] / 1e6).max()
    axl.plot(sc["refL"], sc["refI"] / 1e6, lw=0.9, ls="--", color="0.55",
             label="(0,0) reference")
    axl.plot(lam, spec / 1e6, lw=1.1, color="tab:green", label="this point")
    axl.fill_between(lam, spec / 1e6, color="tab:green", alpha=0.25)
    axl.axvline(4.75, color="tab:red", lw=0.8, ls=":", alpha=0.8)
    mean_l = (lam * spec).sum() / spec.sum() if spec.sum() > 0 else float("nan")
    axl.set_xlim(0, 10)
    axl.set_ylim(0, mlam * 1.08)
    axl.set_xlabel("Wavelength [$\\AA$]")
    axl.set_ylabel("I [$10^6$ n/s]", fontsize=8)
    axl.set_title("chopper band,  $\\langle\\lambda\\rangle$ = %.2f $\\AA$" % mean_l,
                  fontsize=9)
    axl.tick_params(labelsize=8)
    axl.legend(fontsize=7, loc="upper right")
    axl.grid(alpha=.3)

    fig.savefig(os.path.join(folder, "detector.png"), dpi=95,
                bbox_inches="tight")
    plt.close(fig)
    return tag


def make_montage(sc):
    vals = list(range(-20, 21, MONT_STEP))          # -20,-16,...,+20
    n = len(vals)
    fig, axes = plt.subplots(n, n, figsize=(1.35 * n, 1.45 * n))
    for r, ymm in enumerate(reversed(vals)):        # y increases upward
        for c, xmm in enumerate(vals):
            ax = axes[r, c]
            img = load_image(os.path.join(SCAN, name_of(xmm, ymm)))
            ax.imshow(img, origin="lower", extent=EXTENT, cmap="viridis",
                      vmin=0, vmax=sc["vmax"])
            ax.set_xlim(MONT_CROP[0], MONT_CROP[1])
            ax.set_ylim(MONT_CROP[2], MONT_CROP[3])
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 1.02, "%+d,%+d" % (xmm, ymm), fontsize=6,
                    ha="center", va="bottom", transform=ax.transAxes)
            if c == 0:
                ax.set_ylabel("%+d" % ymm, fontsize=8)
            if r == n - 1:
                ax.set_xlabel("%+d" % xmm, fontsize=8)
    fig.suptitle("EQ-SANS detector image vs guide offset (x,y) in mm — every %d mm, "
                 "shared scale, cropped to x%+d..%+d y%+d..%+d cm"
                 % ((MONT_STEP,) + MONT_CROP), fontsize=13)
    fig.supxlabel("guide x offset [mm]")
    fig.supylabel("guide y offset [mm]")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    out = os.path.join(HERE, "detector_montage.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out, "(%dx%d panels)" % (n, n))


if __name__ == "__main__":
    sc = global_scales()
    print("shared scales: vmax=%.4g  proj_x_max=%.4g  proj_y_max=%.4g"
          % (sc["vmax"], sc["px"], sc["py"]))

    if len(sys.argv) < 2 or sys.argv[1] != "montage":
        folders = sorted(glob.glob(os.path.join(SCAN, "x*_y*")))
        folders = [f for f in folders if os.path.exists(os.path.join(f, "detector.dat"))]
        print("rendering %d per-point images ..." % len(folders))
        with Pool() as pool:
            for i, _ in enumerate(pool.imap_unordered(
                    plot_one, [(f, sc) for f in folders], chunksize=8), 1):
                if i % 200 == 0:
                    print("  %d/%d" % (i, len(folders)), flush=True)
        print("per-point images done")

    make_montage(sc)
