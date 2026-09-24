#!/usr/bin/env python3
"""Plots from a saved train_regressor.py checkpoint:

  r2_bar.png            R^2 per target, CNN vs MLP (from the checkpoint's
                         own stored numbers -- always available)
  loss_curves.png        train/val loss vs epoch (only if the checkpoint was
                         saved after the history-tracking fix; older
                         checkpoints -- e.g. an already-completed smoke run --
                         won't have this, and the script says so rather than
                         failing)
  predicted_vs_true.png  scatter per target, both models, on the checkpoint's
                         own recorded test instances (re-inference only, no
                         retraining -- fast, safe to run on a login node)

Usage:
  python3 plot_results.py --checkpoint eqsans_regressor_checkpoint.pt \\
      --manifest submit_smoke_manifest.json --data dataset_smoke
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_regressor import (
    TARGET_NAMES, DetectorCNN, IQMLP, NQ, load_records, ImageDataset, IQDataset, pick_device,
)
from torch.utils.data import DataLoader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="eqsans_regressor_checkpoint.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    names = ckpt["target_names"]
    mean, std = ckpt["target_mean"], ckpt["target_std"]

    print(f"checkpoint: {args.checkpoint}")
    print(f"  train/val/test instances: {len(ckpt['train_instances'])}/"
          f"{len(ckpt['val_instances'])}/{len(ckpt['test_instances'])}")
    print(f"  has loss history: {'history_cnn' in ckpt}")

    # ---------------------------------------------------------- R^2 bars ---
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, [ckpt["r2_cnn"][n] for n in names], w, label="CNN (image)")
    ax.bar(x + w/2, [ckpt["r2_mlp"][n] for n in names], w, label="MLP (I(q))")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R$^2$ (held-out test)")
    ax.set_title("EQ-SANS cylinder regression: test R$^2$ per target")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig("r2_bar.png", dpi=130)
    print("saved r2_bar.png")

    # ------------------------------------------------------- loss curves ---
    if "history_cnn" in ckpt and "history_mlp" in ckpt:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, hist, title in [(ax[0], ckpt["history_cnn"], "CNN (image)"),
                               (ax[1], ckpt["history_mlp"], "MLP (I(q))")]:
            ep = [h["epoch"] for h in hist]
            a.plot(ep, [h["train_loss"] for h in hist], label="train")
            a.plot(ep, [h["val_loss"] for h in hist], label="val")
            a.set_xlabel("epoch"); a.set_ylabel("masked MSE (standardized)")
            a.set_title(title); a.legend(); a.grid(alpha=.3)
        fig.suptitle("Training curves")
        fig.tight_layout()
        fig.savefig("loss_curves.png", dpi=130)
        print("saved loss_curves.png")
    else:
        print("SKIPPED loss_curves.png: this checkpoint predates the history-tracking fix "
              "(it was saved by an older train_regressor.py). Future runs will have it.")

    # -------------------------------------------------- predicted vs true --
    records, missing = load_records(Path(args.manifest), Path(args.data))
    test_ids = set(ckpt["test_instances"])
    test_records = [r for r in records if r["instance_id"] in test_ids]
    print(f"  re-inference on {len(test_records)} test rows ({len(missing)} missing from disk)")

    if not test_records:
        print("SKIPPED predicted_vs_true.png: none of the checkpoint's test instances "
              "were found under --data / --manifest (wrong dataset for this checkpoint?)")
    elif "img_mean" not in ckpt:
        print("SKIPPED predicted_vs_true.png: this checkpoint predates the input-"
              "normalization fix (see train_regressor.py's note on "
              "load_detector_image_raw), so its saved weights were trained against "
              "per-image-max-normalized inputs that this script no longer reproduces. "
              "Re-run training with the current train_regressor.py to get a checkpoint "
              "this plot can use.")
    else:
        cnn = DetectorCNN(len(names)); cnn.load_state_dict(ckpt["cnn_state"]); cnn.to(device).eval()
        mlp = IQMLP(NQ, len(names)); mlp.load_state_dict(ckpt["mlp_state"]); mlp.to(device).eval()

        def predict(model, dataset_cls, norm_mean, norm_std):
            loader = DataLoader(dataset_cls(test_records, norm_mean, norm_std), batch_size=32)
            preds, trues, masks = [], [], []
            with torch.inference_mode():
                for x, y, m in loader:
                    p = model(x.to(device)).cpu().numpy() * std + mean
                    preds.append(p); trues.append(y.numpy()); masks.append(m.numpy())
            return np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)

        p_cnn, y_true, m_true = predict(cnn, ImageDataset, ckpt["img_mean"], ckpt["img_std"])
        p_mlp, _, _ = predict(mlp, IQDataset, ckpt["iq_mean"], ckpt["iq_std"])

        fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4))
        for j, name in enumerate(names):
            v = m_true[:, j] > 0.5
            for row, (preds, label) in enumerate([(p_cnn, "CNN"), (p_mlp, "MLP")]):
                a = axes[row, j]
                if v.sum() < 3:
                    a.set_title(f"{name}\n(no valid test rows)"); a.axis("off"); continue
                yt, yp = y_true[v, j], preds[v, j]
                a.scatter(yt, yp, s=10, alpha=.6)
                lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
                a.plot([lo, hi], [lo, hi], "k--", lw=1)
                r2 = ckpt["r2_cnn" if label == "CNN" else "r2_mlp"][name]
                a.set_title(f"{label} {name}\n$R^2$={r2:.3f}", fontsize=9)
                a.set_xlabel("true"); a.set_ylabel("pred")
                a.grid(alpha=.3)
        fig.suptitle("Predicted vs. true on held-out test instances")
        fig.tight_layout()
        fig.savefig("predicted_vs_true.png", dpi=130)
        print("saved predicted_vs_true.png")
