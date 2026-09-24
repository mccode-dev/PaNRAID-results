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
 
# Each sample: (folder WITH incoherent scattering = input, folder WITHOUT = target)
PAIRS = {
    "cop":   ("cop_inco",   "cop_no_inco"),
    "cupr":  ("cupr_inco",  "cupr_no_inco"),
    "base":  ("with_inco",  "ground_truth"),
    "nal":   ("nal_inco",   "nal_no_inco"),
    "water": ("water", "water_no_inco"),
}
FOLDERS = [name for pair in PAIRS.values() for name in pair]   # all 10 folders
 
START, END   = 0, 100                          # numbered subfolders to load (inclusive)
MONITOR_NAME = "cyl_monitor_tof"                     # replace with your monitor's name
FIG_DIR      = "figures"                         # where the PNG figures are saved
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
 
VAL_FRAC     = 0.15     # NEW: fraction of folders for validation
TEST_FRAC    = 0.15     # fraction of folders for testing
EPOCHS       = 2000
LR           = 1e-3
PATIENCE     = 200      # NEW: stop if validation loss hasn't improved for this many epochs
USE_WEIGHTED = True     # False -> plain MSE (ignores the McStas errors)
 
# Data perturbation (augmentation) - applied to the TRAINING set only
N_AUG          = 5      # perturbed copies per training simulation (0 = off)
NOISE_LEVEL    = 1.0    # Gaussian noise on the input, in units of the McStas error sigma
SCALE_JITTER   = 0.05   # random overall intensity scale factor, 1 +/- 5% (same for input and target)
PERTURB_TARGET = False  # True -> also add noise (sigma = McStas error) to the target
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
    """Fully connected network: flattened I_inc -> flattened I_gt."""
    def __init__(self, n_in, n_out, hidden=(512, 256, 512), dropout=0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
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
 
# --- Perturbations: add noisy / rescaled copies of the training data ---
# Validation and test sets stay untouched so the evaluation uses real simulations only.
if N_AUG > 0:
    S_inc_train = (E_inc.flatten(1) / scale)[train_idx]
    X_orig = X_train.clone()
    Xs, Ys, Ss = [X_train], [Y_train], [S_train]
    for _ in range(N_AUG):
        f  = 1 + SCALE_JITTER * torch.randn(len(train_idx), 1, device=DEVICE)   # one factor per simulation
        Xp = (X_orig + NOISE_LEVEL * S_inc_train * torch.randn_like(X_orig)) * f
        Yp = Y_train * f
        if PERTURB_TARGET:
            Yp = Yp + NOISE_LEVEL * S_train * f * torch.randn_like(Y_train)
        Xs.append(Xp.clamp_min(0))                   # intensities cannot be negative
        Ys.append(Yp.clamp_min(0))
        Ss.append(S_train * f.abs())
    X_train, Y_train, S_train = torch.cat(Xs), torch.cat(Ys), torch.cat(Ss)
    print(f"Perturbation: {N_AUG} copies per simulation -> {X_train.shape[0]} training rows "
          f"(was {len(train_idx)})")
 
    # Example figure: one training simulation and its perturbed copies
    os.makedirs(FIG_DIR, exist_ok=True)
    ex = train_idx[0].item()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(monitor_shape) == 1:
        for j in range(1, N_AUG + 1):
            ax.plot((Xs[j][0] * scale).cpu(), color="gray", alpha=0.4,
                    label="perturbed input" if j == 1 else None)
        ax.plot((X_orig[0] * scale).cpu(), color="C0", label="original input (with inco)")
        ax.plot((Y[ex] * scale).cpu(), color="C1", label="ground truth")
        ax.set_xlabel("Bin"); ax.set_ylabel("Intensity"); ax.legend()
    else:
        im = ax.imshow(((Xs[1][0] - X_orig[0]) * scale).reshape(monitor_shape).cpu(),
                       origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label="perturbed - original")
    ax.set_title(f"Perturbation example: {keys[ex][0]} #{keys[ex][1]}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "perturbation_example.png"), dpi=150)
    plt.close(fig)
 
model = MLP(n_in=X.shape[1], n_out=Y.shape[1]).to(DEVICE)
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
    fig.savefig(out, dpi=500)
    plt.close(fig)
    print(f"Saved {out}")
 