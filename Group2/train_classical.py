#!/usr/bin/env python3
"""Classical regressors (random forest, extra trees, gradient boosting, XGBoost,
SVR, kNN, ridge) on the SAME data and the SAME train/val/test split as
train_regressor.py, so R^2 numbers are directly comparable to the CNN/MLP.

Two feature sets:
  iq       the 60-point log10 I(q) curve (what the MLP sees)
  iq+img   I(q) plus features from the 2D image: log total counts, intensity
           centroid and widths, quadrant flux fractions, and an 8x8 block-summed
           log image (what the CNN can see, in tabular form)

Targets are the 7 from train_regressor.py plus abs_contrast (|contrast|), because
intensity depends on contrast^2 so the sign of contrast is not recoverable.
Sample targets are trained/scored on sample_on=1 rows only; gx/gy on all rows.
Nothing is tuned on the test split.

Usage:
  python3 train_classical.py --manifest submit_full_manifest.json --data dataset_campaign \\
      --nn-checkpoint eqsans_regressor_checkpoint.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_regressor import NQ, QMAX, QMIN, TARGET_NAMES, group_split, load_iq_raw, load_records

TARGETS = TARGET_NAMES + ["abs_contrast"]
SAMPLE_TARGETS = set(TARGETS) - {"gx_mm", "gy_mm"}
BLOCK = 8


# --------------------------------------------------------------- features ---
def extract(path):
    iq = load_iq_raw(path)
    finite = np.isfinite(iq)
    iq = np.where(finite, iq, iq[finite].mean() if finite.any() else 0.0)

    img = np.loadtxt(path / "detector.dat", comments="#")[:256, :]   # rows = y (1.4 m), cols = x (1.0 m)
    tot = max(img.sum(), 1e-30)
    idx = np.arange(256)
    px = img.sum(0) / tot
    py = img.sum(1) / tot
    cx, cy = (px * idx).sum(), (py * idx).sum()
    sx = np.sqrt((px * (idx - cx) ** 2).sum()); sy = np.sqrt((py * (idx - cy) ** 2).sum())
    quad = [img[:128, :128].sum(), img[:128, 128:].sum(), img[128:, :128].sum(), img[128:, 128:].sum()]
    blocks = img.reshape(BLOCK, 256 // BLOCK, BLOCK, 256 // BLOCK).sum(axis=(1, 3))
    feats = np.concatenate([
        iq,
        [np.log10(tot + 1), (cx - 127.5) * 1000 / 256, (cy - 127.5) * 1400 / 256,
         sx * 1000 / 256, sy * 1400 / 256], np.array(quad) / tot,
        np.log10(blocks.ravel() + 1),
    ])
    return feats.astype(np.float32)


def feature_names():
    q = np.logspace(np.log10(QMIN), np.log10(QMAX), NQ)
    return ([f"I(q={v:.3f})" for v in q]
            + ["log10 total counts", "centroid x [mm]", "centroid y [mm]", "width x [mm]", "width y [mm]",
               "quad y-lo/x-lo", "quad y-lo/x-hi", "quad y-hi/x-lo", "quad y-hi/x-hi"]
            + [f"block {i // BLOCK},{i % BLOCK}" for i in range(BLOCK * BLOCK)])


def build_matrices(records, jobs, cache):
    run_ids = [r["run_id"] for r in records]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["run_ids"]) == run_ids:
            print(f"loaded cached features: {cache}")
            X = z["X"]
            return X, *targets_from(records)
    t0 = time.time()
    X = np.stack(Parallel(n_jobs=jobs)(delayed(extract)(r["path"]) for r in records))
    print(f"extracted features for {len(records)} runs in {time.time() - t0:.0f}s")
    np.savez(cache, X=X, run_ids=np.array(run_ids))
    return X, *targets_from(records)


def targets_from(records):
    Y = np.stack([r["targets"] for r in records])
    M = np.stack([r["mask"] for r in records])
    j = TARGET_NAMES.index("contrast")
    Y = np.concatenate([Y, np.abs(Y[:, [j]])], axis=1)
    M = np.concatenate([M, M[:, [j]]], axis=1)
    return Y, M


# ----------------------------------------------------------------- models ---
def model_factories():
    f = {
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "kNN (k=10)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "SVR (RBF)": lambda: TransformedTargetRegressor(
            make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.05)), transformer=StandardScaler()),
        "Random forest": lambda: RandomForestRegressor(300, min_samples_leaf=2, max_features=0.33, n_jobs=-1, random_state=0),
        "Extra trees": lambda: ExtraTreesRegressor(300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=0),
        "Hist grad. boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        f["XGBoost"] = lambda: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                                            colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=0)
    except ImportError:
        print("xgboost not installed: skipping XGBoost")
    return f


def rows_for(target, idx, M):
    j = TARGETS.index(target)
    return np.array([i for i in idx if M[i, j] > 0.5])


# ------------------------------------------------------------------ plots ---
def plot_heatmap(results, nn_r2, path):
    labels, table = [], []
    for name, r2 in nn_r2.items():
        labels.append(name); table.append([r2.get(t, np.nan) for t in TARGETS])
    for fs in results:
        for m, r2 in results[fs].items():
            labels.append(f"{m} [{fs}]"); table.append([r2[t] for t in TARGETS])
    a = np.array(table, dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(TARGETS) + 3.5, 0.42 * len(labels) + 1.6))
    im = ax.imshow(np.clip(a, 0, 1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS, rotation=25, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(a[i, j]) else f"{a[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="test R$^2$ (clipped to 0-1 for colour)")
    ax.set_title("Held-out test R$^2$: neural nets vs classical regressors")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def plot_importance(rf_models, names, path):
    show = ["gx_mm", "gy_mm", "radius", "length", "pd_radius", "abs_contrast"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.ravel(), show):
        imp = rf_models[t].feature_importances_
        top = np.argsort(imp)[::-1][:10][::-1]
        ax.barh([names[i] for i in top], imp[top], color="#2f6f4f")
        ax.set_title(t); ax.tick_params(axis="y", labelsize=8); ax.grid(alpha=.3, axis="x")
    fig.suptitle("Random forest [iq+img]: which features carry each target (top 10)", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def plot_scatter(best, preds, truths, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, t in zip(axes.ravel(), TARGETS):
        m, fs, r2 = best[t]
        yt, yp = truths[t], preds[(fs, m, t)]
        ax.scatter(yt, yp, s=6, alpha=.5, color="#2f6f4f")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{t}\n{m} [{fs}]  R$^2$={r2:.3f}", fontsize=9)
        ax.set_xlabel("true"); ax.set_ylabel("predicted"); ax.grid(alpha=.3)
    fig.suptitle("Best classical model per target, held-out test rows", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved {path}")


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="submit_full_manifest.json")
    ap.add_argument("--data", default="dataset_campaign")
    ap.add_argument("--seed", type=int, default=20260924, help="must match train_regressor.py for the same split")
    ap.add_argument("--jobs", type=int, default=-1, help="joblib workers for feature extraction")
    ap.add_argument("--nn-checkpoint", default=None, help="add CNN/MLP rows to the comparison")
    ap.add_argument("--models", default=None, help="comma list of model names to run (default: all)")
    ap.add_argument("--cache", default="classical_features_cache.npz")
    args = ap.parse_args()

    nn_r2 = {}
    if args.nn_checkpoint:
        if Path(args.nn_checkpoint).exists():
            import torch
            ck = torch.load(args.nn_checkpoint, map_location="cpu", weights_only=False)
            if ck.get("seed") != args.seed:
                raise SystemExit(f"checkpoint seed {ck.get('seed')} != --seed {args.seed}: "
                                 "different train/test split, comparison would be invalid")
            nn_r2 = {"CNN (image) [NN]": ck["r2_cnn"], "MLP (I(q)) [NN]": ck["r2_mlp"]}
        else:
            print(f"note: {args.nn_checkpoint} not found; NN rows will be left out of the comparison")

    records, missing = load_records(Path(args.manifest), Path(args.data))
    print(f"records: {len(records)} complete, {len(missing)} missing")
    if not records:
        raise SystemExit(f"no completed runs found under --data {args.data!r} for manifest {args.manifest!r}. "
                         "The campaign data lives on Juliet; run this there, from the folder that holds the data.")
    idx = group_split(records, seed=args.seed)
    for k, v in idx.items():
        print(f"  {k:5s}: {len(v)} runs")

    X, Y, M = build_matrices(records, args.jobs, Path(args.cache))
    print(f"feature matrix: {X.shape}")
    n_iq = NQ
    feature_sets = {"iq": slice(0, n_iq), "iq+img": slice(0, X.shape[1])}
    factories = model_factories()
    if args.models:
        keep = [m.strip() for m in args.models.split(",")]
        factories = {k: v for k, v in factories.items() if k in keep}

    results = {fs: {m: {} for m in factories} for fs in feature_sets}
    preds, truths, rf_models = {}, {}, {}
    for fs, cols in feature_sets.items():
        for mname, make in factories.items():
            t0 = time.time()
            for t in TARGETS:
                j = TARGETS.index(t)
                tr, te = rows_for(t, idx["train"], M), rows_for(t, idx["test"], M)
                model = make().fit(X[tr][:, cols], Y[tr, j])
                p = model.predict(X[te][:, cols])
                results[fs][mname][t] = float(r2_score(Y[te, j], p))
                preds[(fs, mname, t)] = p; truths[t] = Y[te, j]
                if fs == "iq+img" and mname == "Random forest":
                    rf_models[t] = model
            json.dump({"results": results}, open("classical_results.json", "w"), indent=1)
            print(f"  [{fs:6s}] {mname:20s} {time.time() - t0:5.1f}s  " +
                  "  ".join(f"{t}={results[fs][mname][t]:.2f}" for t in TARGETS))

    print("\n=== held-out test R^2 ===")
    print(f"{'model [features]':<32s}" + "".join(f"{t:>13s}" for t in TARGETS))
    for fs in results:
        for m, r2 in results[fs].items():
            print(f"{m + ' [' + fs + ']':<32s}" + "".join(f"{r2[t]:13.3f}" for t in TARGETS))

    best = {}
    for t in TARGETS:
        cand = [(m, fs, results[fs][m][t]) for fs in results for m in results[fs]]
        best[t] = max(cand, key=lambda c: c[2])

    json.dump({"results": results, "best": {t: list(v) for t, v in best.items()}},
              open("classical_results.json", "w"), indent=1)
    print("saved classical_results.json")
    plot_heatmap(results, nn_r2, "classical_r2.png")
    if rf_models:
        plot_importance(rf_models, feature_names(), "classical_importance.png")
    plot_scatter(best, preds, truths, "classical_predicted_vs_true.png")
