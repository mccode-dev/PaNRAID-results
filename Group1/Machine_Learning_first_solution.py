import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mcstasscript as ms


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
BASE_PATH    = "Group1/daniel"

# Each sample used for TRAINING / VALIDATION / TEST:
# (folder WITH incoherent scattering = input, folder WITHOUT = target)
# cop is deliberately NOT here: it is kept apart for the final prediction test (STEP 9).
PAIRS = {
    "cupr":  ("cupr_inco",  "cupr_no_inco"),
    "base":  ("with_inco",  "ground_truth"),
    "nal":   ("nal_inco",   "nal_no_inco"),
    "water": ("water_inco", "water_no_inco"),
}
FOLDERS = [name for pair in PAIRS.values() for name in pair]   # the 8 training folders

START, END   = 0, 242                          # numbered subfolders to load (inclusive)
MONITOR_NAME = "cyl_monitor_tof"                     # replace with your monitor's name
FIG_DIR      = "figures_sol"                         # where the PNG figures are saved
MODEL_FILE   = "trained_model_sol.pt"                # trained model is saved here (reuse with Predict.py)

# Prediction test on data the model has NEVER seen (not in training, validation or test):
# PREDICT_INC  = simulation(s) WITH incoherent -> the model predicts the one WITHOUT.
#                Either one simulation folder (e.g. ".../cop_inco/12") or a folder of
#                numbered simulations (".../cop_inco") to predict all of them.
# PREDICT_TRUE = matching McStas result WITHOUT incoherent, used only to compare
#                (set to None if you don't have it). Set PREDICT_INC = None to skip.
PREDICT_INC  = "Group1/daniel/cop_inco"
PREDICT_TRUE = "Group1/daniel/cop_no_inco"
GIF_FRAME_MS = 300      # time each prediction figure is shown in the GIF (milliseconds)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

VAL_FRAC     = 0.15     # NEW: fraction of folders for validation
TEST_FRAC    = 0.15     # fraction of folders for testing
EPOCHS       = 2000
LR           = 1e-3
PATIENCE     = 200      # NEW: stop if validation loss hasn't improved for this many epochs
BG_KNOTS     = 40       # background smoothness: number of points it is defined at (None = every bin)
USE_WEIGHTED = True     # False -> plain MSE (ignores the McStas errors)

SEED         = 0


# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------
def load_folders(base_path, start=0, end=20):
    """Load McStas data from numbered subfolders base_path/start ... base_path/end."""
    data = {}
    for i in range(start, end + 1):
        folder = os.path.join(base_path, str(i))
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue
        data[i] = ms.load_data(folder)
        print(f"Loaded {folder}")
    return data


def to_tensor(data_dict, monitor_name, keys, field="Intensity"):
    """Stack one monitor's array from each folder into a tensor of shape (N_folders, *monitor_shape)."""
    arrays = []
    for i in keys:
        monitor = ms.name_search(monitor_name, data_dict[i])
        arrays.append(np.asarray(getattr(monitor, field), dtype=np.float32))
    return torch.from_numpy(np.stack(arrays))


def weighted_mse(pred, target, sigma):
    """Chi-squared-style loss: squared residuals divided by the McStas variance."""
    return torch.mean(((pred - target) / sigma) ** 2)


def plain_mse(pred, target, sigma=None):
    return torch.mean((pred - target) ** 2)


loss_fn = weighted_mse if USE_WEIGHTED else plain_mse


# ---------------------------------------------------------------------
# STEP 1: load McStas simulations
# ---------------------------------------------------------------------
raw = {name: load_folders(os.path.join(BASE_PATH, name), START, END) for name in FOLDERS}

for name in FOLDERS:
    print(f"{name:15s}: {len(raw[name])} simulations loaded")

first = next((raw[n] for n in FOLDERS if raw[n]), None)
if first is None:
    raise RuntimeError(f"No data found in any folder under {BASE_PATH}.")
print("Available monitors:", [d.name for d in first[min(first)]])


# ---------------------------------------------------------------------
# STEP 2: convert every folder to PyTorch tensors (intensity + error)
# ---------------------------------------------------------------------
# tensors[folder_name] = {"keys": [...], "I": tensor, "E": tensor}
# Within each pair only simulation numbers present in BOTH folders are kept,
# so input and target rows line up one-to-one.
tensors = {}
for sample, (inc_name, gt_name) in PAIRS.items():
    common = sorted(set(raw[inc_name]) & set(raw[gt_name]))
    only_one = set(raw[inc_name]) ^ set(raw[gt_name])
    if only_one:
        print(f"[{sample}] folders present in only one of {inc_name}/{gt_name} (ignored): {sorted(only_one)}")
    if not common:
        print(f"[{sample}] no matching simulations -> sample skipped")
        continue
    for name in (inc_name, gt_name):
        tensors[name] = {
            "keys": common,
            "I": to_tensor(raw[name], MONITOR_NAME, common, "Intensity"),
            "E": to_tensor(raw[name], MONITOR_NAME, common, "Error"),
        }
        print(f"{name:15s}: I {tuple(tensors[name]['I'].shape)}  E {tuple(tensors[name]['E'].shape)}")

used = [s for s, (inc, gt) in PAIRS.items() if inc in tensors]
if not used:
    raise RuntimeError("No sample has matching data in both its folders.")

shapes = {tuple(tensors[n]["I"].shape[1:]) for n in tensors}
if len(shapes) > 1:
    raise RuntimeError(f"Monitor shapes differ between folders: {shapes}")

# Combine all samples into one dataset: input = *_inco, target = *_no_inco / ground_truth
I_inc = torch.cat([tensors[PAIRS[s][0]]["I"] for s in used]).to(DEVICE)
E_inc = torch.cat([tensors[PAIRS[s][0]]["E"] for s in used]).to(DEVICE)
I_gt  = torch.cat([tensors[PAIRS[s][1]]["I"] for s in used]).to(DEVICE)
E_gt  = torch.cat([tensors[PAIRS[s][1]]["E"] for s in used]).to(DEVICE)

# keys[i] = (sample, simulation number) for row i of the tensors above
keys = [(s, k) for s in used for k in tensors[PAIRS[s][0]]["keys"]]

print(f"I_gt: {tuple(I_gt.shape)}   I_inc: {tuple(I_inc.shape)}   device: {DEVICE}")


# ---------------------------------------------------------------------
# STEP 3: multilayer perceptron (MLP) + data split
# ---------------------------------------------------------------------
class MLP(nn.Module):
    """Background-subtraction network.

    The network predicts ONLY the incoherent background B(x) and returns  x - B(x).
      * B is smooth (defined at bg_knots points, interpolated in between).
      * B >= 0 (softplus): the model can only remove intensity, never add it,
        so it cannot invent new peaks.
      * B <= x: the result never goes below zero.
      * At the start of training B is ~0, so the model begins as "output = input"
        and only learns how much background to take away.
    """
    def __init__(self, n_in, n_out, hidden=(512, 256, 512), dropout=0.1, bg_knots=40):
        super().__init__()
        self.n_out, self.bg_knots = n_out, bg_knots
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        # The background is predicted at only `bg_knots` points and linearly interpolated
        # to all bins -> it is always smooth and cannot cut into sharp peaks.
        last = nn.Linear(prev, bg_knots if bg_knots else n_out)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -6.0)      # softplus(-6) ~ 0.0025 -> background starts ~0
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def background(self, x):
        """Predicted incoherent background (same units as x), between 0 and x."""
        b = nn.functional.softplus(self.net(x))
        if self.bg_knots:
            b = nn.functional.interpolate(b.unsqueeze(1), size=self.n_out,
                                          mode="linear", align_corners=True).squeeze(1)
        return torch.minimum(b, x.clamp_min(0))

    def forward(self, x):
        return x - self.background(x)


# --- NEW: train / validation / test split ---
torch.manual_seed(SEED)
N       = len(keys)
perm    = torch.randperm(N)
n_test  = max(1, round(TEST_FRAC * N))
n_val   = max(1, round(VAL_FRAC * N))
n_train = N - n_test - n_val
if n_train < 1:
    raise RuntimeError(f"Not enough folders ({N}) for a train/val/test split.")

test_idx  = perm[:n_test]
val_idx   = perm[n_test:n_test + n_val]
train_idx = perm[n_test + n_val:]

print(f"Train folders ({n_train}): {[keys[i] for i in train_idx.tolist()]}")
print(f"Val folders   ({n_val}): {[keys[i] for i in val_idx.tolist()]}")
print(f"Test folders  ({n_test}): {[keys[i] for i in test_idx.tolist()]}")

test_idx, val_idx, train_idx = test_idx.to(DEVICE), val_idx.to(DEVICE), train_idx.to(DEVICE)

# --- Flatten and normalise (scale from training data only) ---
monitor_shape = I_gt.shape[1:]
scale = I_inc[train_idx].abs().max()

X = I_inc.flatten(1) / scale
Y = I_gt.flatten(1)  / scale
S = E_gt.flatten(1)  / scale

S_train_pos = S[train_idx][S[train_idx] > 0]
S = S.clamp_min(S_train_pos.min() if S_train_pos.numel() else 1e-6)

X_train, Y_train, S_train = X[train_idx], Y[train_idx], S[train_idx]
X_val,   Y_val,   S_val   = X[val_idx],   Y[val_idx],   S[val_idx]      # NEW
X_test,  Y_test,  S_test  = X[test_idx],  Y[test_idx],  S[test_idx]


model = MLP(n_in=X.shape[1], n_out=Y.shape[1], bg_knots=BG_KNOTS).to(DEVICE)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------
# STEP 4: training (with validation after every epoch)
# ---------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
best_val, best_epoch = float("inf"), 0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(1, EPOCHS + 1):
    # --- training step ---
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), Y_train, S_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # --- NEW: validation step (no weight updates) ---
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), Y_val, S_val).item()
    val_losses.append(val_loss)

    # --- NEW: keep the best model ---
    if val_loss < best_val:
        best_val, best_epoch = val_loss, epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:5d} | train {loss.item():.4e} | val {val_loss:.4e}")

    # --- NEW: early stopping ---
    if epoch - best_epoch >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

model.load_state_dict(best_state)
print(f"Best validation loss {best_val:.4e} at epoch {best_epoch} -> restored this model")

plt.figure()
plt.semilogy(train_losses, label="training")
plt.semilogy(val_losses,   label="validation")
plt.axvline(best_epoch - 1, color="gray", ls="--", label=f"best epoch ({best_epoch})")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and validation loss")
plt.legend(); plt.show()


# ---------------------------------------------------------------------
# STEP 5: validation summary
# ---------------------------------------------------------------------
model.eval()
with torch.no_grad():
    Y_val_pred = model(X_val)
    print(f"Validation chi2/bin -> model: {weighted_mse(Y_val_pred, Y_val, S_val).item():.4e} "
          f"| baseline (I_inc): {weighted_mse(X_val, Y_val, S_val).item():.4e}")


# ---------------------------------------------------------------------
# STEP 6: testing (used only once, at the very end)
# ---------------------------------------------------------------------
with torch.no_grad():
    Y_pred = model(X_test)
    chi2_model    = weighted_mse(Y_pred, Y_test, S_test).item()
    chi2_baseline = weighted_mse(X_test, Y_test, S_test).item()
    mse_model     = plain_mse(Y_pred, Y_test).item()
    mse_baseline  = plain_mse(X_test, Y_test).item()

print(f"Test chi2/bin -> model: {chi2_model:.4e} | baseline (I_inc): {chi2_baseline:.4e}")
print(f"Test MSE      -> model: {mse_model:.4e} | baseline (I_inc): {mse_baseline:.4e}")

# Back to physical units and original shape
pred_I = (Y_pred * scale).reshape(-1, *monitor_shape).cpu()
true_I = I_gt[test_idx].cpu()
inc_I  = I_inc[test_idx].cpu()
true_E = E_gt[test_idx].cpu()

# ---------------------------------------------------------------------
# STEP 7: one figure per sample, saved as PNG
# ---------------------------------------------------------------------
os.makedirs(FIG_DIR, exist_ok=True)

split_of = {}
for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
    for i in idx.tolist():
        split_of[i] = name

with torch.no_grad():
    all_pred = (model(X) * scale).reshape(-1, *monitor_shape).cpu()
all_inc, all_gt, all_E = I_inc.cpu(), I_gt.cpu(), E_gt.cpu()

for sample in used:
    rows = [i for i, (s, _) in enumerate(keys) if s == sample]
    # Prefer a test simulation (unseen by the model), then validation, then training
    row = next((i for split in ("test", "val", "train") for i in rows if split_of[i] == split), None)
    if row is None:
        continue
    num, split = keys[row][1], split_of[row]
    title = f"{sample} - simulation #{num} ({split} set)"
    out = os.path.join(FIG_DIR, f"{sample}_sim{num}_{split}.png")

    if len(monitor_shape) == 1:
        x = np.arange(monitor_shape[0])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, all_inc[row], label="with incoherent (input)", alpha=0.6)
        ax.errorbar(x, all_gt[row], yerr=all_E[row], fmt=".", label="ground truth ± σ")
        ax.plot(x, all_pred[row], label="MLP prediction")
        ax.set_xlabel("Bin"); ax.set_ylabel("Intensity")
        ax.legend(); ax.set_title(title)
    else:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for a, img, t in zip(ax, [all_inc[row], all_gt[row], all_pred[row]],
                             ["Input (with inco)", "Ground truth", "MLP prediction"]):
            im = a.imshow(img, origin="lower", aspect="auto")
            a.set_title(t); fig.colorbar(im, ax=a)
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------
# STEP 8: save the trained model
# ---------------------------------------------------------------------
torch.save({
    "state_dict":    model.state_dict(),
    "n_in":          X.shape[1],
    "n_out":         Y.shape[1],
    "scale":         scale.item(),
    "monitor_shape": tuple(monitor_shape),
    "monitor_name":  MONITOR_NAME,
    "model_type":    "background_subtraction",
    "bg_knots":      BG_KNOTS,
}, MODEL_FILE)
print(f"Saved trained model to {MODEL_FILE}")


# ---------------------------------------------------------------------
# STEP 9: prediction on unseen simulations (cop): with incoherent -> without
# ---------------------------------------------------------------------
def read_monitor(sim_path, field="Intensity"):
    monitor = ms.name_search(MONITOR_NAME, ms.load_data(sim_path))
    return torch.from_numpy(np.asarray(getattr(monitor, field), dtype=np.float32))


def remove_incoherent(I_in):
    """Model prediction WITHOUT incoherent scattering for one input monitor (with incoherent).
    Returns (prediction, predicted incoherent background); prediction = input - background."""
    if tuple(I_in.shape) != tuple(monitor_shape):
        raise RuntimeError(f"Monitor shape {tuple(I_in.shape)} does not match the "
                           f"training data {tuple(monitor_shape)}.")
    model.eval()
    with torch.no_grad():
        x = (I_in.flatten() / scale).unsqueeze(0).to(DEVICE)
        I_out = (model(x) * scale).reshape(monitor_shape).cpu()
        B     = (model.background(x) * scale).reshape(monitor_shape).cpu()
    return I_out, B


def make_gif(folder, out_file, pattern="prediction_", duration_ms=GIF_FRAME_MS):
    """Combine folder/prediction_*.png into one animated GIF (numeric order)."""
    from PIL import Image
    import re
    files = [f for f in os.listdir(folder) if f.startswith(pattern) and f.endswith(".png")]
    if not files:
        print(f"No {pattern}*.png files in {folder}, no GIF made")
        return
    num = lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)]
    files.sort(key=num)
    frames = [Image.open(os.path.join(folder, f)).convert("RGB") for f in files]
    frames[0].save(out_file, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"Saved {out_file} ({len(frames)} frames)")


def numbered_subfolders(path):
    return sorted(int(d) for d in os.listdir(path)
                  if d.isdigit() and os.path.isdir(os.path.join(path, d)))


if PREDICT_INC and os.path.isdir(PREDICT_INC):
    pred_dir = os.path.join(FIG_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # One simulation, or every numbered simulation in the folder
    nums = numbered_subfolders(PREDICT_INC)
    jobs = ([(str(n), os.path.join(PREDICT_INC, str(n)),
              os.path.join(PREDICT_TRUE, str(n)) if PREDICT_TRUE else None) for n in nums]
            if nums else [("_".join(os.path.normpath(PREDICT_INC).split(os.sep)[-2:]), PREDICT_INC, PREDICT_TRUE)])

    predictions, mse_model, mse_base = {}, [], []
    for name, inc_path, true_path in jobs:
        I_in  = read_monitor(inc_path)
        I_out, B = remove_incoherent(I_in)
        predictions[name] = I_out
        has_true = true_path is not None and os.path.isdir(true_path)
        if has_true:
            I_true, E_true = read_monitor(true_path), read_monitor(true_path, "Error")
            mse_model.append(torch.mean((I_out - I_true) ** 2).item())
            mse_base.append(torch.mean((I_in - I_true) ** 2).item())

        # Figure: input, prediction and (if available) true no-incoherent McStas result
        if len(monitor_shape) == 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(I_in, label="McStas with incoherent (input)", alpha=0.6)
            if has_true:
                ax.errorbar(np.arange(monitor_shape[0]), I_true, yerr=E_true, fmt=".",
                            label="McStas without incoherent (true) ± σ")
            ax.plot(I_out, label="ML prediction without incoherent")
            ax.plot(B, color="gray", ls="--", label="predicted incoherent background")
            ax.set_xlabel("Bin"); ax.set_ylabel("Intensity"); ax.legend()
        else:
            imgs, titles = [I_in, I_out, B], ["Input (with inco)", "ML prediction (no inco)", "Predicted background"]
            if has_true:
                imgs.append(I_true); titles.append("True (no inco)")
            fig, ax = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 4))
            for a, img, t in zip(ax, imgs, titles):
                im = a.imshow(img, origin="lower", aspect="auto"); a.set_title(t); fig.colorbar(im, ax=a)
        fig.suptitle(f"Prediction (unseen data): {os.path.basename(os.path.normpath(PREDICT_INC))} {name}"
                     if nums else f"Prediction (unseen data): {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(pred_dir, f"prediction_{name}.png"), dpi=150)
        plt.close(fig)

    # All predictions as one tensor (rows in the order of the simulation numbers) + saved to disk
    pred_names = list(predictions)
    I_no_inco  = torch.stack([predictions[n] for n in pred_names])
    torch.save({"names": pred_names, "prediction": I_no_inco},
               os.path.join(pred_dir, "predictions_no_inco.pt"))
    print(f"Predicted {len(pred_names)} simulation(s) -> tensor {tuple(I_no_inco.shape)}, "
          f"figures in {pred_dir}")

    # Animated GIF of all prediction figures (sorted by simulation number)
    make_gif(pred_dir, os.path.join(pred_dir, "predictions.gif"))

    if mse_model:
        print(f"Unseen-data MSE -> model: {np.mean(mse_model):.4e} | "
              f"baseline (input with inco): {np.mean(mse_base):.4e}  "
              f"({len(mse_model)} simulations compared with the true no-incoherent result)")
elif PREDICT_INC:
    print(f"PREDICT_INC not found, skipping prediction: {PREDICT_INC}")
