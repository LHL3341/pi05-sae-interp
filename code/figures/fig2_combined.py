"""Figure 1 (a, c, d) + Figure 2 (b saliency) for the paper.

Two-figure layout:
  Figure 1 — top row: (a) steering strip plot, (c) CV histogram;
            bottom row: (d) activation heatmap across episodes
  Figure 2 — (b) saliency hero (high-CV / low-CV), 2 × 5 grid
"""
from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from sae import TopKSAE  # noqa: E402

OUT = ROOT / "figures"

FEATURES = [7676, 2036, 1102]
ALPHAS = [-3.0, -1.0, 0.0, 1.0, 3.0]
HIGH_CV_FEAT = 2319
LOW_CV_FEAT  = 6951
N_HERO_COLS = 5

# ---------- load steering data ----------
RUNS = ROOT / "runs"
def deltas_for(run_dir):
    rows = []
    for f in FEATURES:
        base = np.load(run_dir / f"feat{f}_alpha+0.00.npy")
        norms = []
        for a in ALPHAS:
            d = np.load(run_dir / f"feat{f}_alpha{a:+.2f}.npy")
            norms.append(np.linalg.norm(d - base))
        rows.append(norms)
    return np.array(rows)
trained_sae   = deltas_for(RUNS / "steer_real_v2")
trained_rand  = deltas_for(RUNS / "steer_real_random_dir")
untrained_sae = deltas_for(RUNS / "steer_random_libero")

# ---------- compute SAE codes for (c) and (d) ----------
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
codes_FT = codes.mean(axis=(2, 3))
mean_T = codes_FT.mean(axis=1); std_T = codes_FT.std(axis=1)
cv_FT = std_T / np.maximum(mean_T, 1e-6)
fire_per_frame = codes_FT.max(axis=1) > 0
fire_rate = fire_per_frame.mean(axis=0)
keep = fire_rate > 0.02
med_cv = np.full(cv_FT.shape[1], np.nan)
for f in range(cv_FT.shape[1]):
    if keep[f] and fire_per_frame[:, f].any():
        med_cv[f] = np.median(cv_FT[fire_per_frame[:, f], f])
med_cv_kept = med_cv[~np.isnan(med_cv)]

# ---------- load LIBERO meta for (d) ----------
parquets = sorted((ROOT / "datasets/libero_10_image/data/chunk-000").glob("file-*.parquet"))
dfs = [pq.read_table(p).to_pandas() for p in parquets]
full = pd.concat(dfs, ignore_index=True)
eps = sorted(full["episode_index"].unique().tolist())
per_ep = max(1, n_frames // len(eps))
chunks_d = []
for ep in eps:
    sub = full[full["episode_index"] == ep]
    n_ep = min(per_ep, len(sub))
    if n_ep == 0: continue
    step = max(1, len(sub) // n_ep)
    chunks_d.append(sub.iloc[::step].iloc[:n_ep])
df = pd.concat(chunks_d, ignore_index=True).iloc[:n_frames].reset_index(drop=True)
ep_per_frame = df["episode_index"].values

index = json.loads((OUT / "feature_browser_v2/index.json").read_text())
high = index[str(HIGH_CV_FEAT)]
low  = index[str(LOW_CV_FEAT)]

# ---------- prepare (d) heatmap matrix ----------
per_frame = codes.max(axis=(1, 2, 3))
fire_per_frame_d = per_frame > 0
fire_rate_d = fire_per_frame_d.mean(axis=0)
peak_d = per_frame.max(axis=0)
fire_ok = (fire_rate_d > 0.05) & (fire_rate_d < 0.95)
low_cv_idx  = np.where(fire_ok & (med_cv < 0.2))[0]
high_cv_idx = np.where(fire_ok & (med_cv > 1.0))[0]
low_cv_idx  = sorted(low_cv_idx,  key=lambda f: -peak_d[f])[:10]
high_cv_idx = sorted(high_cv_idx, key=lambda f: -peak_d[f])[:10]
ordered = list(low_cv_idx) + list(high_cv_idx)
ord_frames = np.argsort(np.lexsort((np.arange(n_frames), ep_per_frame)))
mat = per_frame[np.ix_(ord_frames, ordered)].T

# ============================================================
# Figure 1: (a) steering | (c) CV hist  ;  (d) activation heatmap
# ============================================================
fig1 = plt.figure(figsize=(7.0, 3.7), constrained_layout=False)
outer1 = GridSpec(2, 1, figure=fig1, height_ratios=[1.05, 0.85],
                  hspace=0.85, left=0.075, right=0.96, top=0.93, bottom=0.10)
top1 = GridSpecFromSubplotSpec(1, 2, outer1[0, 0], wspace=0.30,
                                width_ratios=[1.0, 1.05])

# (a) steering strip
ax_a = fig1.add_subplot(top1[0, 0])
mask_alpha = np.array([a != 0.0 for a in ALPHAS])
groups = [
    ("trained\nSAE dir",   trained_sae[:,  mask_alpha].ravel(), "C0"),
    ("trained\nrand dir",  trained_rand[:, mask_alpha].ravel(), "C2"),
    ("untrained\nSAE dir", untrained_sae[:,mask_alpha].ravel(), "C3"),
]
rng = np.random.default_rng(0)
for i, (label, vals, color) in enumerate(groups):
    jitter = rng.uniform(-0.10, 0.10, len(vals))
    ax_a.scatter(np.full_like(vals, i) + jitter, vals,
                 s=14, color=color, alpha=0.65, edgecolor="none")
    ax_a.hlines(np.median(vals), i - 0.30, i + 0.30, color=color, lw=2.0)
ax_a.set_xticks(range(len(groups)))
ax_a.set_xticklabels([g[0] for g in groups], fontsize=7)
ax_a.set_ylabel(r"$\|\Delta a\|_2$", fontsize=8)
ax_a.set_yscale("log")
ax_a.set_ylim(0.1, 30)
ax_a.set_title("(a) steering response: trained saturates 25$\\times$ below random",
               fontsize=8, loc="left")
ax_a.tick_params(labelsize=7)
ax_a.grid(axis="y", which="major", alpha=0.3, lw=0.4)
ax_a.annotate("", xy=(2, 13), xytext=(2, 0.5),
              arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8))
ax_a.text(2.18, 2.5, r"$\sim25\times$", fontsize=7.5, color="0.2", ha="left", va="center")

# (c) CV hist
ax_c = fig1.add_subplot(top1[0, 1])
bins = np.linspace(0, 3, 50)
ax_c.hist(med_cv_kept, bins=bins, color="C0", alpha=0.85, edgecolor="white", lw=0.4)
ax_c.axvline(0.5, color="C3", ls="--", lw=0.8)
ax_c.set_xlabel("median CV across denoise steps", fontsize=8)
ax_c.set_ylabel("# features", fontsize=8)
ax_c.set_title("(c) feature stability", fontsize=8, loc="left")
ax_c.tick_params(labelsize=7)
n_low = int((med_cv_kept < 0.5).sum())
n_high = int((med_cv_kept >= 0.5).sum())
ax_c.text(0.45, 0.85, f"low-CV: {n_low}",
          transform=ax_c.transAxes, ha="right", fontsize=7, color="C0")
ax_c.text(0.95, 0.85, f"high-CV: {n_high}",
          transform=ax_c.transAxes, ha="right", fontsize=7, color="C3")

# (d) activation heatmap (bottom row)
ax_d = fig1.add_subplot(outer1[1, 0])
cmap_d = LinearSegmentedColormap.from_list("act", ["white", "#0c5fc7", "#000033"])
vmax = max(1.0, np.percentile(mat[mat > 0], 99)) if (mat > 0).any() else 1.0
im = ax_d.imshow(mat, aspect="auto", cmap=cmap_d, vmin=0, vmax=vmax,
                 interpolation="nearest")
ep_sorted = ep_per_frame[ord_frames]
prev = ep_sorted[0]
for i, e in enumerate(ep_sorted[1:], 1):
    if e != prev:
        ax_d.axvline(i - 0.5, color="0.4", lw=0.5); prev = e
ax_d.axhline(len(low_cv_idx) - 0.5, color="C3", lw=1.0)
ax_d.set_yticks([len(low_cv_idx)/2 - 0.5,
                 len(low_cv_idx) + len(high_cv_idx)/2 - 0.5])
ax_d.set_yticklabels(["low-CV", "high-CV"], fontsize=7, rotation=90, va="center")
ax_d.tick_params(axis="x", labelsize=6)

boundary = [0]
for i in range(1, len(ep_sorted)):
    if ep_sorted[i] != ep_sorted[i-1]:
        boundary.append(i)
boundary.append(len(ep_sorted))
mids = [(boundary[i] + boundary[i+1]) // 2 for i in range(len(boundary)-1)]
ax_d.set_xticks(mids)
ax_d.set_xticklabels([f"ep{ep_sorted[m]}" for m in mids], fontsize=6)
ax_d.set_title("(d) activation heatmap of top-10 low-CV / top-10 high-CV features (rows) across episode-sorted frames (cols)",
               fontsize=7.5, loc="left", pad=4)

# colorbar inside the figure on the right
pos = ax_d.get_position()
cax = fig1.add_axes([pos.x1 + 0.005, pos.y0, 0.010, pos.height])
fig1.colorbar(im, cax=cax).ax.tick_params(labelsize=6)

fig1.savefig(OUT / "fig1.pdf", dpi=300)
fig1.savefig(OUT / "fig1.png", dpi=300)
print(f"saved {OUT/'fig1.pdf'}")

# ============================================================
# Figure 2: (b) saliency tiles — 2 rows × 5 cols, separate figure
# ============================================================
fig2 = plt.figure(figsize=(7.0, 2.6), constrained_layout=False)
gs2 = GridSpec(2, N_HERO_COLS, figure=fig2,
               hspace=0.30, wspace=0.04,
               left=0.10, right=0.99, top=0.84, bottom=0.06)

TILES = OUT / "saliency_tiles"
def hero_row(row_axes, info, label, color, feat):
    top = info["top_frames"][:N_HERO_COLS]
    vals = info["top_values"][:N_HERO_COLS]
    tids = info["top_task_ids"][:N_HERO_COLS]
    for ax, idx, v, tid in zip(row_axes, top, vals, tids):
        tile = TILES / f"feat{feat}_f{idx}.png"
        img = np.array(Image.open(tile).convert("RGB")) if tile.exists() else \
              np.array(Image.open(io.BytesIO(df.iloc[idx]["observation.images.image"]["bytes"])).convert("RGB"))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"a={v:.1f}", fontsize=6.5, color=color, pad=1)
        ax.set_xlabel(f"f{idx}·t{tid}", fontsize=6)
    row_axes[0].set_ylabel(label, fontsize=6.8, color=color, rotation=90, labelpad=2)

axes_high = [fig2.add_subplot(gs2[0, i]) for i in range(N_HERO_COLS)]
axes_low  = [fig2.add_subplot(gs2[1, i]) for i in range(N_HERO_COLS)]
hero_row(axes_high, high,
         f"feat {HIGH_CV_FEAT}\nhigh-CV={high['median_cv']:.2f}\ntask-specific\nmotion phase",
         color="C3", feat=HIGH_CV_FEAT)
hero_row(axes_low, low,
         f"feat {LOW_CV_FEAT}\nlow-CV={low['median_cv']:.2f}\ncross-task\nsemantic",
         color="C0", feat=LOW_CV_FEAT)

fig2.text(0.10, 0.94,
          "(b) top-activating LIBERO frames per feature  "
          "(overlay = $|\\nabla\\,$feat$\\,\\cdot\\,$image$|$)",
          fontsize=8, ha="left", va="bottom")

fig2.savefig(OUT / "fig2_saliency.pdf", dpi=300)
fig2.savefig(OUT / "fig2_saliency.png", dpi=300)
print(f"saved {OUT/'fig2_saliency.pdf'}")
