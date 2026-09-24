#!/usr/bin/env python3
"""Train and compare two PyTorch regressors on the EQ-SANS cylinder campaign:

  - DetectorCNN : 2D detector image -> 7 targets   (sees anisotropy from misalignment)
  - IQMLP       : 1D I(q) curve     -> 7 targets   (radially averaged, no anisotropy)

Targets: gx_mm, gy_mm, radius, length, pd_radius, pd_length, contrast.
Sample-parameter targets are masked out for sample_on=0 rows (no sample
present); gx/gy are always valid (alignment is an instrument property).

Follows this workshop's own training conventions from
23_September_Wednesday/06_morning_samples_1/SANS_inverse_problem/02_PyTorch_training.ipynb:
device selection (cuda/mps/cpu), lazy per-file loading in __getitem__,
reproducible seeded splits, and a checkpoint that carries metadata, not just
weights. Adapted from that notebook's classification setup to multi-target
regression with a masked loss.

Usage:
  python3 train_regressor.py --manifest manifest.json --data dataset_campaign
  python3 train_regressor.py --manifest manifest_smoke.json --data dataset_smoke --epochs 5
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TARGET_NAMES = ["gx_mm", "gy_mm", "radius", "length", "pd_radius", "pd_length", "contrast"]
IMG_NY = IMG_NX = 256
NQ = 60
QMIN, QMAX = 0.003, 0.15


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------- records ---
def load_records(manifest_path, data_root):
    manifest = json.load(open(manifest_path))
    records, missing = [], []
    for row in manifest:
        d = data_root / row["run_id"]
        if (d / "mccode.sim").exists():
            targets = np.array([row[t] if t not in ("gx_mm", "gy_mm") else row[t]
                                 for t in TARGET_NAMES], dtype=np.float32)
            mask = np.array([row["targets_valid"][t] for t in TARGET_NAMES], dtype=np.float32)
            records.append(dict(path=d, instance_id=row["instance_id"],
                                run_id=row["run_id"], targets=targets, mask=mask))
        else:
            missing.append(row["run_id"])
    return records, missing


def group_split(records, fractions=(0.70, 0.15, 0.15), seed=0):
    """Split by instance_id so paired sample_on=0/1 rows never cross splits."""
    instances = sorted({r["instance_id"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(instances)
    n = len(instances)
    n_tr = int(n * fractions[0]); n_va = int(n * fractions[1])
    groups = {"train": set(instances[:n_tr]),
              "val": set(instances[n_tr:n_tr + n_va]),
              "test": set(instances[n_tr + n_va:])}
    idx = {k: [i for i, r in enumerate(records) if r["instance_id"] in v] for k, v in groups.items()}
    for k in idx:
        rng.shuffle(idx[k])
    return idx


# --------------------------------------------------------------- loaders ---
# NOTE on normalization: earlier versions of this file divided each image by
# its OWN max ("img / img.max()"). That erases absolute intensity by
# construction -- every image gets rescaled to peak at 1.0 regardless of
# whether it started at 1e6 or 1e18 counts. Since `contrast` sets scattered
# intensity via (sld-sld_solvent)^2, that per-sample normalization was
# actively deleting the one signal that target lives in (confirmed: a real
# smoke-test run got contrast R^2 = 0.01-0.06, near zero, while radius --
# a shape-encoded target unaffected by this bug -- got R^2 = 0.74-0.84 on
# the same data). Fixed by normalizing with statistics computed ONCE from
# the training split and applied uniformly, so relative differences between
# samples survive.
def load_detector_image_raw(path):
    raw = np.loadtxt(path / "detector.dat", comments="#")
    img = raw[:IMG_NY, :]                       # first block = intensity
    return np.log1p(np.clip(img, 0, None)).astype(np.float32)


def load_iq_raw(path, qgrid=np.logspace(np.log10(QMIN), np.log10(QMAX), NQ)):
    d = np.loadtxt(path / "qdet.dat", comments="#")
    q, I = d[:, 0], d[:, 1]
    g = (q > 0) & (I > 0)
    if g.sum() < 5:
        return np.full(NQ, np.nan, dtype=np.float32)
    Ii = np.interp(qgrid, q[g], I[g])
    return np.log10(np.maximum(Ii, 1e-30)).astype(np.float32)


def fit_normalization(records, loader_fn, sample_size=200, seed=0):
    """Mean/std of the raw (log-space) representation over a random subset
    of records -- a fixed, dataset-level affine transform, not per-sample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(sample_size, len(records)), replace=False)
    vals = np.stack([loader_fn(records[i]["path"]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), float(max(vals.std(), 1e-6))


class ImageDataset(Dataset):
    def __init__(self, records, mean=0.0, std=1.0):
        self.records = records
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = (load_detector_image_raw(r["path"]) - self.mean) / self.std
        return img[None, :, :], r["targets"], r["mask"]


class IQDataset(Dataset):
    def __init__(self, records, mean=0.0, std=1.0):
        self.records = records
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        iq = (load_iq_raw(r["path"]) - self.mean) / self.std
        return np.nan_to_num(iq, nan=0.0), r["targets"], r["mask"]


# ---------------------------------------------------------------- models ---
class DetectorCNN(nn.Module):
    """Same backbone as the workshop's classification DetectorCNN, linear
    regression head (n_targets outputs, no activation) instead of softmax."""
    def __init__(self, n_targets):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_targets),
        )

    def forward(self, x):
        return self.head(self.features(x))


class IQMLP(nn.Module):
    def __init__(self, n_in, n_targets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------- training --
def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return se.sum() / denom


def run_epoch(model, loader, device, mean, std, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            yn = (y - mean) / std
            pred = model(x)
            loss = masked_mse(pred, yn, m)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
    return total_loss / max(total_n, 1)


def evaluate_r2(model, loader, device, mean, std):
    model.eval()
    preds, trues, masks = [], [], []
    with torch.inference_mode():
        for x, y, m in loader:
            pred = model(x.to(device)).cpu().numpy() * std.numpy() + mean.numpy()
            preds.append(pred); trues.append(y.numpy()); masks.append(m.numpy())
    preds, trues, masks = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
    r2 = {}
    for j, name in enumerate(TARGET_NAMES):
        v = masks[:, j] > 0.5
        if v.sum() < 5:
            r2[name] = float("nan"); continue
        yt, yp = trues[v, j], preds[v, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2[name] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2


def train_one(model, train_loader, val_loader, test_loader, device, mean, std, epochs, lr=1e-3):
    model = model.to(device)
    mean_t, std_t = mean.to(device), std.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state = float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device, mean_t, std_t, optimizer)
        va = run_epoch(model, val_loader, device, mean_t, std_t)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val:
            best_val, best_state = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:3d}  train_loss={tr:.4f}  val_loss={va:.4f}")
    model.load_state_dict(best_state)
    return model, evaluate_r2(model, test_loader, device, mean, std), history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--data", default="dataset_campaign")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"PyTorch {torch.__version__}, device: {device}")

    records, missing = load_records(Path(args.manifest), Path(args.data))
    print(f"records: {len(records)} complete, {len(missing)} missing")

    idx = group_split(records, seed=args.seed)
    for k, v in idx.items():
        print(f"  {k:5s}: {len(v)} runs, {len({records[i]['instance_id'] for i in v})} instances")

    # standardize targets from the TRAINING split only, masked
    train_targets = np.stack([records[i]["targets"] for i in idx["train"]])
    train_masks = np.stack([records[i]["mask"] for i in idx["train"]])
    mean = np.array([np.average(train_targets[:, j], weights=np.maximum(train_masks[:, j], 1e-6))
                     for j in range(len(TARGET_NAMES))], dtype=np.float32)
    std = np.array([np.sqrt(np.average((train_targets[:, j] - mean[j]) ** 2,
                                       weights=np.maximum(train_masks[:, j], 1e-6)))
                    for j in range(len(TARGET_NAMES))], dtype=np.float32)
    std = np.maximum(std, 1e-6)
    mean_t, std_t = torch.tensor(mean), torch.tensor(std)

    num_workers = 4 if device.type == "cuda" else 0
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=num_workers,
                         persistent_workers=num_workers > 0)
    results = {}

    # Input normalization, fit ONCE from the training split (not per-sample --
    # see the note above load_detector_image_raw for why per-sample
    # normalization was actively destroying the `contrast` signal).
    train_records = [records[i] for i in idx["train"]]
    img_mean, img_std = fit_normalization(train_records, load_detector_image_raw)
    iq_mean, iq_std = fit_normalization(train_records, load_iq_raw)
    print(f"image input norm: mean={img_mean:.3f} std={img_std:.3f}")
    print(f"I(q) input norm:  mean={iq_mean:.3f} std={iq_std:.3f}")

    common_meta = dict(
        target_names=TARGET_NAMES, target_mean=mean, target_std=std,
        seed=args.seed, manifest=str(args.manifest),
        train_instances=sorted({records[i]["instance_id"] for i in idx["train"]}),
        val_instances=sorted({records[i]["instance_id"] for i in idx["val"]}),
        test_instances=sorted({records[i]["instance_id"] for i in idx["test"]}),
        img_mean=img_mean, img_std=img_std, iq_mean=iq_mean, iq_std=iq_std,
    )

    print("\n=== DetectorCNN (2D image input) ===")
    cnn_loaders = {k: DataLoader(ImageDataset([records[i] for i in idx[k]], img_mean, img_std),
                                 shuffle=(k == "train"), **loader_kwargs) for k in idx}
    cnn = DetectorCNN(len(TARGET_NAMES))
    cnn, r2_cnn, hist_cnn = train_one(cnn, cnn_loaders["train"], cnn_loaders["val"], cnn_loaders["test"],
                                      device, mean_t, std_t, args.epochs)
    results["cnn"] = r2_cnn

    # Saved immediately, not just at the very end: train_one() for the MLP is
    # about to run for the same number of epochs again, and if the SLURM
    # --time budget is exceeded partway through THAT, this is what keeps a
    # fully-trained CNN from being silently lost (the only other save call is
    # after both models finish -- see the epoch-count discussion this was
    # added for).
    torch.save(dict(common_meta, cnn_state=cnn.state_dict(), r2_cnn=r2_cnn, history_cnn=hist_cnn),
              "eqsans_regressor_checkpoint_cnn_only.pt")
    print("saved eqsans_regressor_checkpoint_cnn_only.pt (insurance in case MLP training doesn't finish)")

    print("\n=== IQMLP (1D I(q) input) ===")
    iq_loaders = {k: DataLoader(IQDataset([records[i] for i in idx[k]], iq_mean, iq_std),
                                shuffle=(k == "train"), **loader_kwargs) for k in idx}
    mlp = IQMLP(NQ, len(TARGET_NAMES))
    mlp, r2_mlp, hist_mlp = train_one(mlp, iq_loaders["train"], iq_loaders["val"], iq_loaders["test"],
                                      device, mean_t, std_t, args.epochs)
    results["mlp"] = r2_mlp

    print("\n=== R^2 on held-out test instances ===")
    print(f"{'target':<12s} {'CNN (image)':>12s} {'MLP (I(q))':>12s}")
    for name in TARGET_NAMES:
        print(f"{name:<12s} {r2_cnn[name]:12.3f} {r2_mlp[name]:12.3f}")

    torch.save(dict(common_meta, cnn_state=cnn.state_dict(), mlp_state=mlp.state_dict(),
                    r2_cnn=r2_cnn, r2_mlp=r2_mlp, history_cnn=hist_cnn, history_mlp=hist_mlp),
              "eqsans_regressor_checkpoint.pt")
    print("\nsaved eqsans_regressor_checkpoint.pt")
