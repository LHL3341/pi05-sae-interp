"""Figure 2(c): per-feature variability across flow-matching denoise steps.

The proposal's Aim 3 hypothesis: in a continuous-time vector field, semantic
features should remain near-constant across denoise steps for a given frame
(they describe the scene), while physical/dynamic features should vary with
the time-step (they describe motion). The coefficient of variation
``cv = std/mean`` across denoise steps, computed per (feature, frame, batch),
should therefore be bimodal-ish: a low-CV cluster (semantic) and a higher-CV
cluster (physical).

We compute CV per (feature, frame) pair and plot the per-feature 25th/50th/75th
percentile across frames. A low-CV stable feature has all three percentiles
near zero; a dynamic feature has high spread.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from sae import TopKSAE  # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Load SAE + activations.
ckpt = torch.load(ROOT / "runs/sae_real/sae.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
sae = TopKSAE(cfg["d_in"], cfg["d_dict"], cfg["k"], cfg["k_aux"])
sae.load_state_dict(ckpt["state_dict"])
sae.eval()

acts = np.load(ROOT / "activations/real_pi05_libero/pre_action_proj.npy")
# shape: [num_calls, B, action_horizon, D] = [2560, 1, 10, 1024]
# num_calls = num_frames * num_flow_steps = 256 * 10 = 2560
n_frames, n_flow = 256, 10
acts = acts.reshape(n_frames, n_flow, *acts.shape[1:])  # [F=256, T=10, B=1, H=10, D=1024]

# Encode all activations
with torch.no_grad():
    flat = torch.from_numpy(acts.reshape(-1, acts.shape[-1])).float()
    codes_chunks = []
    for i in range(0, flat.shape[0], 8192):
        z = sae.encode(flat[i:i+8192])
        codes_chunks.append(sae.topk(z, sae.k))
    codes = torch.cat(codes_chunks, dim=0).numpy()
codes = codes.reshape(n_frames, n_flow, 1, 10, -1)  # [F, T, B, H, d_dict]

# Average over batch and action_horizon → [F, T, d_dict]
codes_FT = codes.mean(axis=(2, 3))  # [F, T, d_dict]

# CV across denoise step T, per (frame, feature)
mean_T = codes_FT.mean(axis=1)  # [F, d_dict]
std_T  = codes_FT.std(axis=1)   # [F, d_dict]
cv_FT = std_T / np.maximum(mean_T, 1e-6)  # [F, d_dict]

# Restrict to features that actually fire on a reasonable fraction of frames
fire_per_frame = (codes_FT.max(axis=1) > 0)  # [F, d_dict]
fire_rate = fire_per_frame.mean(axis=0)  # [d_dict]
keep = fire_rate > 0.02  # at least 2% of frames
print(f"features fired on >2% frames: {keep.sum()}/{len(keep)}")

# Median CV (across frames where this feature fires) per feature
med_cv = []
for f in range(cv_FT.shape[1]):
    if not keep[f]: continue
    vals = cv_FT[fire_per_frame[:, f], f]
    if len(vals) == 0: continue
    med_cv.append(np.median(vals))
med_cv = np.asarray(med_cv)
print(f"median CV percentiles: 10%={np.percentile(med_cv,10):.3f} 50%={np.percentile(med_cv,50):.3f} 90%={np.percentile(med_cv,90):.3f}")

fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
bins = np.linspace(0, 3, 60)
ax.hist(med_cv, bins=bins, color="C0", alpha=0.85, edgecolor="white", lw=0.4)
ax.axvline(0.5, color="C3", ls="--", lw=0.8, label="CV=0.5")
ax.set_xlabel("median CV across denoise steps")
ax.set_ylabel("# features")
ax.set_title("Aim 3: feature stability across flow-matching steps", fontsize=8)
ax.legend(fontsize=7, frameon=False)

# Annotate
n_low = (med_cv < 0.5).sum()
n_high = (med_cv >= 0.5).sum()
ax.annotate(f"low-CV ({n_low})\n≈ semantic candidates",
            xy=(0.4, 0.92), xycoords="axes fraction",
            ha="right", va="top", fontsize=7, color="C0")
ax.annotate(f"high-CV ({n_high})\n≈ flow-time-varying",
            xy=(0.55, 0.92), xycoords="axes fraction",
            ha="left", va="top", fontsize=7, color="C3")

fig.savefig(OUT / "fig2c_cv.pdf", dpi=300)
fig.savefig(OUT / "fig2c_cv.png", dpi=300)
print(f"saved {OUT/'fig2c_cv.pdf'}")
