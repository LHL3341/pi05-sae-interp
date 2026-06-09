"""Activation heatmap [feature × frame], sorted by CV.

Visualises the semantic / physical decoupling more directly than the CV
histogram: low-CV features (top of heat map) light up on scattered frames
across episodes, while high-CV features (bottom) light up on contiguous
bands of frames within a single episode.

Optionally also produce a [feature × denoise-step] heatmap for a small set of
example features to expose flow-time variation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from sae import TopKSAE  # noqa: E402

OUT = ROOT / "figures"

# ---- load ----
ckpt = torch.load(ROOT / "runs/sae_real_v2/sae.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
sae = TopKSAE(cfg["d_in"], cfg["d_dict"], cfg["k"], cfg["k_aux"])
sae.load_state_dict(ckpt["state_dict"]); sae.eval()
acts = np.load(ROOT / "activations/real_pi05_libero_v2/pre_action_proj.npy")
n_frames, n_flow, B, H, D = 480, 10, 1, 10, 1024
acts = acts.reshape(n_frames, n_flow, B, H, D)
with torch.no_grad():
    flat = torch.from_numpy(acts.reshape(-1, D)).float()
    chunks = []
    for i in range(0, flat.shape[0], 8192):
        z = sae.encode(flat[i:i+8192])
        chunks.append(sae.topk(z, sae.k))
    codes = torch.cat(chunks, dim=0).numpy()
codes = codes.reshape(n_frames, n_flow, B, H, -1)

# Per-frame feature peak (max over denoise steps × action tokens).
per_frame = codes.max(axis=(1, 2, 3))  # [F, d_dict]
fire_per_frame = per_frame > 0
fire_rate = fire_per_frame.mean(axis=0)
codes_FT = codes.mean(axis=(2, 3))
mean_T = codes_FT.mean(axis=1)
std_T  = codes_FT.std(axis=1)
cv_FT = std_T / np.maximum(mean_T, 1e-6)
med_cv = np.full(cv_FT.shape[1], np.nan)
for j in range(cv_FT.shape[1]):
    if fire_per_frame[:, j].any():
        med_cv[j] = np.median(cv_FT[fire_per_frame[:, j], j])

# Episode index per frame so the x-axis can be grouped meaningfully.
import pandas as pd
import pyarrow.parquet as pq
parquets = sorted((ROOT / "datasets/libero_10_image/data/chunk-000").glob("file-*.parquet"))
dfs = [pq.read_table(p).to_pandas() for p in parquets]
full = pd.concat(dfs, ignore_index=True)
eps = sorted(full["episode_index"].unique().tolist())
per_ep = max(1, n_frames // len(eps))
chunks2 = []
for ep in eps:
    sub = full[full["episode_index"] == ep]
    n_ep = min(per_ep, len(sub))
    if n_ep == 0: continue
    step = max(1, len(sub) // n_ep)
    chunks2.append(sub.iloc[::step].iloc[:n_ep])
df = pd.concat(chunks2, ignore_index=True).iloc[:n_frames].reset_index(drop=True)
ep_per_frame = df["episode_index"].values
task_per_frame = df["task_index"].values

# ---- Pick top-N features by peak activation, in 2 groups ----
fire_ok = (fire_rate > 0.05) & (fire_rate < 0.95)
peak = per_frame.max(axis=0)
low_cv  = np.where(fire_ok & (med_cv < 0.2))[0]
high_cv = np.where(fire_ok & (med_cv > 1.0))[0]
low_cv  = sorted(low_cv,  key=lambda f: -peak[f])[:15]
high_cv = sorted(high_cv, key=lambda f: -peak[f])[:15]
ordered = list(low_cv) + list(high_cv)
labels = ["low-CV"] * len(low_cv) + ["high-CV"] * len(high_cv)

# Sort frames by (episode, frame_in_episode) so episodes appear as contiguous strips.
ord_frames = np.argsort(np.lexsort((np.arange(n_frames), ep_per_frame)))
mat = per_frame[np.ix_(ord_frames, ordered)].T  # [n_features, n_frames]

# ---- plot ----
cmap = LinearSegmentedColormap.from_list("act", ["white", "#0c5fc7", "#000033"])
fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
vmax = max(1.0, np.percentile(mat[mat > 0], 99))
im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
               interpolation="nearest")
# Episode boundary verticals
ep_sorted = ep_per_frame[ord_frames]
prev = ep_sorted[0]
for i, e in enumerate(ep_sorted[1:], 1):
    if e != prev:
        ax.axvline(i - 0.5, color="0.4", lw=0.6)
        prev = e
# Mark cv-group separator
ax.axhline(len(low_cv) - 0.5, color="C3", lw=1.0)

ax.set_yticks(range(len(ordered)))
ax.set_yticklabels([f"{f}" for f in ordered], fontsize=6)
ax.set_ylabel("feature index (low-CV: top  /  high-CV: bottom)", fontsize=8)
ax.set_xlabel("frame index (sorted by episode)", fontsize=8)
ax.set_title("Activation heatmap: low-CV features fire scattered across episodes; high-CV features fire in contiguous bands",
             fontsize=8.5)

# Episode labels along x
boundary = [0]
for i in range(1, len(ep_sorted)):
    if ep_sorted[i] != ep_sorted[i-1]:
        boundary.append(i)
boundary.append(len(ep_sorted))
mids = [(boundary[i] + boundary[i+1]) // 2 for i in range(len(boundary)-1)]
ax.set_xticks(mids)
ax.set_xticklabels([f"ep {ep_sorted[m]}\nt{task_per_frame[ord_frames][m]}" for m in mids], fontsize=6)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("peak activation", fontsize=7)
cbar.ax.tick_params(labelsize=6)

fig.savefig(OUT / "fig_heatmap.pdf", dpi=300)
fig.savefig(OUT / "fig_heatmap.png", dpi=300)
print(f"saved {OUT/'fig_heatmap.pdf'}")
print(f"low-CV features:  {low_cv[:8]} ...")
print(f"high-CV features: {high_cv[:8]} ...")
