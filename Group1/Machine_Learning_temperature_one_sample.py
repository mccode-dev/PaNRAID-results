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
PATH_GT      = "Group1/daniel/ground_truth"   # without incoherent scattering
PATH_INC     = "Group1/daniel/with_inco"      # with incoherent scattering
START, END   = 0, 242                          # folder numbers to load (inclusive)
MONITOR_NAME = "cyl_monitor_tof"                     # replace with your monitor's name
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

VAL_FRAC     = 0.15     # NEW: fraction of folders for validation
TEST_FRAC    = 0.15     # fraction of folders for testing
EPOCHS       = 2000
LR           = 1e-3
PATIENCE     = 200      # NEW: stop if validation loss hasn't improved for this many epochs
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
data_gt  = load_folders(PATH_GT,  START, END)
data_inc = load_folders(PATH_INC, START, END)

keys = sorted(set(data_gt) & set(data_inc))
only_one = set(data_gt) ^ set(data_inc)
if only_one:
    print(f"Folders present in only one dataset (ignored): {sorted(only_one)}")
if not keys:
    raise RuntimeError("No folders found in both datasets.")

print("Available monitors:", [d.name for d in data_gt[keys[0]]])


# ---------------------------------------------------------------------
# STEP 2: convert to PyTorch tensors (intensity + error)
# ---------------------------------------------------------------------
I_gt  = to_tensor(data_gt,  MONITOR_NAME, keys, "Intensity").to(DEVICE)
E_gt  = to_tensor(data_gt,  MONITOR_NAME, keys, "Error").to(DEVICE)
I_inc = to_tensor(data_inc, MONITOR_NAME, keys, "Intensity").to(DEVICE)
E_inc = to_tensor(data_inc, MONITOR_NAME, keys, "Error").to(DEVICE)

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

k = 0
folder = keys[test_idx[k].item()]
if len(monitor_shape) == 1:
    x = np.arange(monitor_shape[0])
    plt.figure()
    plt.plot(x, inc_I[k], label="with incoherent (input)", alpha=0.6)
    plt.errorbar(x, true_I[k], yerr=true_E[k], fmt=".", label="ground truth ± σ")
    plt.plot(x, pred_I[k], label="MLP prediction")
    plt.legend(); plt.title(f"Test folder {folder}")
    plt.show()
else:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, img, title in zip(ax, [inc_I[k], true_I[k], pred_I[k]],
                             ["Input (with inco)", "Ground truth", "MLP prediction"]):
        im = a.imshow(img, origin="lower", aspect="auto")
        a.set_title(title); fig.colorbar(im, ax=a)
    fig.suptitle(f"Test folder {folder}")
    plt.show()