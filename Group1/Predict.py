"""
Use the trained model on ONE new McStas simulation WITH incoherent scattering
and get the prediction WITHOUT incoherent scattering - no retraining needed.

First run Machine_Learning.py once (it saves trained_model.pt), then:
    python Predict.py                       # uses NEW_SIM_PATH below
    python Predict.py path/to/simulation    # or give the folder on the command line
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mcstasscript as ms

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
MODEL_FILE   = "trained_model.pt"
NEW_SIM_PATH = "Group1/daniel/cop_inco/0"         # folder of ONE McStas simulation WITH incoherent
OUT_DIR      = "figures"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# Must be identical to the MLP class in Machine_Learning.py
class MLP(nn.Module):
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


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
model = MLP(ckpt["n_in"], ckpt["n_out"]).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
scale, monitor_shape, monitor_name = ckpt["scale"], tuple(ckpt["monitor_shape"]), ckpt["monitor_name"]


# ---------------------------------------------------------------------
# Load the new simulation and predict
# ---------------------------------------------------------------------
sim_path = sys.argv[1] if len(sys.argv) > 1 else NEW_SIM_PATH
monitor = ms.name_search(monitor_name, ms.load_data(sim_path))
I_in = torch.from_numpy(np.asarray(monitor.Intensity, dtype=np.float32))
if tuple(I_in.shape) != monitor_shape:
    raise RuntimeError(f"Monitor shape {tuple(I_in.shape)} does not match the training data {monitor_shape}.")

with torch.no_grad():
    x = (I_in.flatten() / scale).unsqueeze(0).to(DEVICE)
    I_out = (model(x) * scale).reshape(monitor_shape).cpu().clamp_min(0)   # predicted, no incoherent

print(f"Prediction done: tensor of shape {tuple(I_out.shape)}")


# ---------------------------------------------------------------------
# Save data + figure
# ---------------------------------------------------------------------
name = os.path.basename(os.path.normpath(sim_path))
os.makedirs(OUT_DIR, exist_ok=True)

dat = os.path.join(OUT_DIR, f"{name}_predicted_no_inco.dat")
np.savetxt(dat, I_out.numpy().reshape(I_out.shape[0], -1),
           header="Predicted intensity without incoherent scattering")

if len(monitor_shape) == 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(I_in,  label="McStas with incoherent (input)", alpha=0.6)
    ax.plot(I_out, label="ML prediction without incoherent")
    ax.set_xlabel("Bin"); ax.set_ylabel("Intensity"); ax.legend()
else:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, img, t in zip(ax, [I_in, I_out], ["Input (with inco)", "ML prediction (no inco)"]):
        im = a.imshow(img, origin="lower", aspect="auto"); a.set_title(t); fig.colorbar(im, ax=a)
fig.suptitle(f"New simulation: {name}")
fig.tight_layout()
png = os.path.join(OUT_DIR, f"{name}_predicted_no_inco.png")
fig.savefig(png, dpi=150)
plt.close(fig)

print(f"Saved {png}")
print(f"Saved {dat}")
