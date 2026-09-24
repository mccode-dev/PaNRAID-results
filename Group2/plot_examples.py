#!/usr/bin/env python3
"""Show the variation the regressors train on, using real McStas output.

Re-simulates locally with ./eqsans_cylinder.out (ncount=2e6, ~4.5 s/run) and plots:

  examples_iq_sweeps.png       I(q) when ONE parameter changes from a baseline
  examples_detector_grid.png   the matching 2D detector images
  examples_target_distributions.png  histograms of the 7 targets over the manifest
  examples_gallery.png         random instances taken from the real campaign
                               manifest (true parameters, fresh Monte Carlo
                               seed) + I(q) overlays coloured by radius/contrast

Usage:
  python3 plot_examples.py [--manifest submit_full_manifest.json] [--data-dir examples_data]
"""
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm, Normalize

BINARY = "./eqsans_cylinder.out"
SEED = 12345  # same seed for every controlled run -> differences are physics, not noise
BASE = dict(gx=0.0, gy=0.0, sample_on=1, radius=60.0, length=500.0,
            pd_radius=0.1, pd_length=0.1, sld=4.5, sld_solvent=2.0)

# group -> [(label, overrides)]; every case differs from BASE in one thing only.
GROUPS = {
    "sample":    [("sample off", dict(sample_on=0)), ("sample on", {})],
    "radius":    [(f"radius {r} Å", dict(radius=r)) for r in (20, 60, 120, 200)],
    "length":    [(f"length {l} Å", dict(length=l)) for l in (200, 500, 1000)],
    "pd_radius": [(f"pd_radius {p}", dict(pd_radius=p)) for p in (0.0, 0.15, 0.30)],
    "contrast":  [(f"contrast {4.5 - s:+.1f}", dict(sld_solvent=s)) for s in (0.0, 2.0, 4.0)],
    "sign":      [("contrast +2.5 (sld 4.5, solvent 2.0)", {}),
                  ("contrast -2.5 (sld 1.41, solvent 3.91)", dict(sld=1.41, sld_solvent=3.91))],
    "misalign":  [("aligned (0, 0) mm", {}), ("gx = 10 mm", dict(gx=0.010)),
                  ("gy = 10 mm", dict(gy=0.010)), ("gx = gy = 20 mm", dict(gx=0.020, gy=0.020))],
}


def run_one(outdir, params, ncount, seed=None):
    if (outdir / "qdet.dat").exists():
        return
    cmd = [BINARY, f"--ncount={ncount}", f"--dir={outdir}", "--format=McStas"]
    if seed is not None:
        cmd.append(f"--seed={seed}")
    cmd += [f"{k}={v}" for k, v in params.items()] + ["model_scale=1", "model_abs=0"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_many(jobs, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: run_one(*j), jobs))


def load_image(d):
    return np.loadtxt(d / "detector.dat", comments="#")[:256, :]


def load_iq(d):
    a = np.loadtxt(d / "qdet.dat", comments="#")
    g = (a[:, 0] > 0) & (a[:, 1] > 0)
    return a[g, 0], a[g, 1]


def show_image(ax, img, norm):
    # detector.dat rows are y (1.4 m), columns are x (1.0 m): gx shifts the column
    # centroid, gy the row centroid, and the scattering blob is round in metres.
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(cmap(0.0))          # zero-count pixels -> background colour, not white
    ax.imshow(np.ma.masked_less_equal(img, 0), origin="lower", norm=norm,
              cmap=cmap, extent=[-0.5, 0.5, -0.7, 0.7], aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])


def centroid_mm(img):
    tot = img.sum()
    r = (img.sum(1) * np.arange(256)).sum() / tot
    c = (img.sum(0) * np.arange(256)).sum() / tot
    return (c - 127.5) * 1000 / 256, (r - 127.5) * 1400 / 256   # (x, y) in mm


def controlled_figures(data_dir, ncount, workers, out_prefix):
    jobs, dirs = [], {}
    for g, cases in GROUPS.items():
        for label, ov in cases:
            key = json.dumps({**BASE, **ov}, sort_keys=True)
            d = data_dir / ("c_" + hashlib.md5(key.encode()).hexdigest()[:10])
            dirs[(g, label)] = d
            jobs.append((d, {**BASE, **ov}, ncount, SEED))
    uniq = {str(j[0]): j for j in jobs}
    print(f"controlled cases: {len(uniq)} unique runs")
    run_many(list(uniq.values()), workers)

    # ------------------------------------------------ I(q) sweeps (log-log) ---
    panels = [("radius", "radius (length 500 Å fixed)"), ("length", "length (radius 60 Å fixed)"),
              ("pd_radius", "radius polydispersity"), ("contrast", "contrast = sld - sld_solvent"),
              ("sign", "sign of contrast (same seed)"), ("sample", "sample off vs on")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (g, title) in zip(axes.ravel(), panels):
        cmap = plt.get_cmap("viridis")
        n = len(GROUPS[g])
        for i, (label, _) in enumerate(GROUPS[g]):
            q, I = load_iq(dirs[(g, label)])
            style = dict(ls="--", lw=2.4) if (g == "sign" and i == 1) else dict(lw=1.8)
            ax.loglog(q, I, color=cmap(0.1 + 0.75 * i / max(n - 1, 1)), label=label, **style)
        ax.set_title(title); ax.set_xlabel("q [1/Å]"); ax.set_ylabel("I(q) [arb.]")
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("I(q): what each parameter does to the radially averaged curve "
                 "(one factor changed at a time, ncount = 2e6)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_iq_sweeps.png", dpi=130); plt.close(fig)
    print(f"saved {out_prefix}_iq_sweeps.png")

    # ----------------------------------------------- detector image grid -----
    rows = ["sample", "radius", "length", "pd_radius", "contrast", "misalign"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10.5, 3.4 * len(rows)))
    for r, g in enumerate(rows):
        imgs = [load_image(dirs[(g, lab)]) for lab, _ in GROUPS[g]]
        vmax = max(im.max() for im in imgs)
        norm = LogNorm(vmin=vmax * 1e-6, vmax=vmax)   # shared within a row -> intensity differences stay visible
        for c in range(4):
            ax = axes[r, c]
            if c >= len(imgs):
                ax.axis("off"); continue
            show_image(ax, imgs[c], norm)
            title = GROUPS[g][c][0]
            if g == "misalign":
                (x0, y0), (x1, y1) = centroid_mm(imgs[0]), centroid_mm(imgs[c])
                title += f"\nflux {imgs[c].sum() / imgs[0].sum():.2f}x, centroid shift ({x1 - x0:+.1f}, {y1 - y0:+.1f}) mm"
            ax.set_title(title, fontsize=8.5)
    fig.suptitle("Detector images (log intensity, colour scale shared within each row; "
                 "x horizontal 1.0 m, y vertical 1.4 m)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f"{out_prefix}_detector_grid.png", dpi=110); plt.close(fig)
    print(f"saved {out_prefix}_detector_grid.png")


def gallery_figure(manifest, data_dir, ncount, workers, n_iq, n_img, out_prefix):
    rows = [r for r in json.load(open(manifest)) if r["sample_on"] == 1]
    rng = np.random.default_rng(1)
    pick = [rows[i] for i in rng.choice(len(rows), size=n_iq, replace=False)]
    jobs = []
    for r in pick:
        p = dict(gx=r["gx_mm"] / 1000, gy=r["gy_mm"] / 1000, sample_on=1, radius=r["radius"],
                 length=r["length"], pd_radius=r["pd_radius"], pd_length=r["pd_length"],
                 sld=r["sld_material"], sld_solvent=r["sld_solvent"])
        jobs.append((data_dir / ("g_" + r["run_id"]), p, ncount, None))
    print(f"gallery: {len(jobs)} runs from {manifest}")
    run_many(jobs, workers)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15])
    for k in range(n_img):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        r, d = pick[k], jobs[k][0]
        img = load_image(d)
        show_image(ax, img, LogNorm(vmin=img.max() * 1e-6, vmax=img.max()))
        ax.set_title(f"R={r['radius']:.0f} Å  L={r['length']:.0f} Å  c={r['contrast']:+.1f}\n"
                     f"gx={r['gx_mm']:+.0f} gy={r['gy_mm']:+.0f} mm  pd=({r['pd_radius']:.2f},{r['pd_length']:.2f})",
                     fontsize=8.5)
    for j, (key, label, cmap, norm) in enumerate([
            ("radius", "radius [Å]", "viridis", LogNorm(20, 200)),
            ("abs_contrast", "|contrast|", "plasma", Normalize(0, 9))]):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        cm = plt.get_cmap(cmap)
        for r, (d, *_) in zip(pick, jobs):
            q, I = load_iq(d)
            val = r["radius"] if key == "radius" else abs(r["contrast"])
            ax.loglog(q, I, color=cm(norm(val)), alpha=.7, lw=1.1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        fig.colorbar(sm, ax=ax, label=label)
        ax.set_xlabel("q [1/Å]"); ax.set_ylabel("I(q) [arb.]"); ax.grid(alpha=.3, which="both")
        ax.set_title(f"{n_iq} random campaign instances, coloured by {label}")
    fig.suptitle("Random instances from the real campaign design (true parameters, re-simulated locally; "
                 "each image on its own log colour scale)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_gallery.png", dpi=110); plt.close(fig)
    print(f"saved {out_prefix}_gallery.png")


def distribution_figure(manifest, out_prefix):
    rows = [r for r in json.load(open(manifest)) if r["sample_on"] == 1]
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    hists = [("gx_mm", "gx_mm [mm]", False), ("gy_mm", "gy_mm [mm]", False),
             ("radius", "radius [Å] (log axis)", True), ("length", "length [Å] (log axis)", True),
             ("pd_radius", "pd_radius", False), ("pd_length", "pd_length", False),
             ("contrast", "contrast = sld - sld_solvent", False)]
    for ax, (k, label, log) in zip(axes.ravel(), hists):
        v = col(k)
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 30) if log else 30
        ax.hist(v, bins=bins, color="#2f6f4f", edgecolor="white")
        if log:
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200] if k == "radius" else [200, 400, 700, 1000])
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(label); ax.set_ylabel("instances"); ax.grid(alpha=.3)
    ax = axes[1, 3]
    ax.scatter(col("gx_mm"), col("gy_mm"), s=4, alpha=.5, color="#2f6f4f")
    ax.set_xlabel("gx_mm"); ax.set_ylabel("gy_mm"); ax.grid(alpha=.3)
    ax.set_title(f"misalignment plane ({int(col('aligned').sum())} exactly aligned at origin)", fontsize=9)
    fig.suptitle(f"Training-set distribution of the 7 regression targets ({len(rows)} instances)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_target_distributions.png", dpi=120); plt.close(fig)
    print(f"saved {out_prefix}_target_distributions.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="submit_full_manifest.json")
    ap.add_argument("--data-dir", default="examples_data")
    ap.add_argument("--ncount", default="2e6")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-gallery", type=int, default=40, help="instances for the I(q) overlay")
    args = ap.parse_args()
    if not Path(BINARY).exists():
        raise SystemExit(f"{BINARY} not found: compile with `mcrun -c -n0 --format=McStas eqsans_cylinder.instr` first")
    data_dir = Path(args.data_dir); data_dir.mkdir(exist_ok=True)
    controlled_figures(data_dir, args.ncount, args.jobs, "examples")
    distribution_figure(args.manifest, "examples")
    gallery_figure(args.manifest, data_dir, args.ncount, args.jobs, args.n_gallery, 8, "examples")
