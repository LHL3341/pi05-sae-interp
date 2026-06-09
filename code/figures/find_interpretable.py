"""Find candidate-interpretable SAE features by pulling top-activating frames.

For each LIBERO frame in our 256-frame extraction, we already have all SAE
codes for the (denoise step × action token) grid. To get a per-frame feature
strength, we average codes over (denoise step × action token) — this is the
"frame-level feature activation". Then for each feature we pick the K frames
with the highest activation and dump them as a contact sheet.

Output: ``figures/feature_browser/``  containing
  • ``index.json`` — for each feature: ranked frames, their activations,
                      task ids, fire rate, CV stability score
  • ``feat_{id}.png`` — a 1×K contact sheet (real LIBERO RGB) for that feature
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from sae import TopKSAE  # noqa: E402

OUT = ROOT / "figures" / "feature_browser_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load SAE + activations ----
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
codes = codes.reshape(n_frames, n_flow, B, H, -1)  # [F, T, 1, 10, d_dict]

# Per-frame feature activation: max over denoise step + horizon (peak strength)
per_frame = codes.max(axis=(1, 2, 3))  # [F, d_dict]
fire_per_frame = per_frame > 0
fire_rate = fire_per_frame.mean(axis=0)  # [d_dict]

# CV across denoise steps (same as fig2c)
codes_FT = codes.mean(axis=(2, 3))  # [F, T, d_dict]
mean_T = codes_FT.mean(axis=1)
std_T  = codes_FT.std(axis=1)
cv_FT = std_T / np.maximum(mean_T, 1e-6)
med_cv = np.array([np.median(cv_FT[fire_per_frame[:, f], f]) if fire_per_frame[:, f].any() else np.nan
                   for f in range(cv_FT.shape[1])])

# ---- Load LIBERO images for the 256 frames we extracted ----
tasks_path = ROOT / "datasets/libero_10_image/meta/tasks.parquet"
# Replicate the same stratified sampling that extract_real used.
import sys
sys.path.insert(0, str(ROOT))
from extract_real import load_libero_episode  # noqa: E402
parquets = sorted((ROOT / "datasets/libero_10_image/data/chunk-000").glob("file-*.parquet"))
import pandas as _pd
dfs = [pq.read_table(p).to_pandas() for p in parquets]
full = _pd.concat(dfs, ignore_index=True)
eps = sorted(full["episode_index"].unique().tolist())
per_ep = max(1, n_frames // len(eps))
chunks = []
for ep in eps:
    sub = full[full["episode_index"] == ep]
    n_ep = min(per_ep, len(sub))
    if n_ep == 0: continue
    step = max(1, len(sub) // n_ep)
    chunks.append(sub.iloc[::step].iloc[:n_ep])
df = _pd.concat(chunks, ignore_index=True).iloc[:n_frames]
tasks_df = pq.read_table(tasks_path).to_pandas()
task_id_to_text = dict(zip(tasks_df["task_index"], tasks_df.index)) \
    if tasks_df.index.name == "task" else dict(zip(tasks_df.index, tasks_df["task"]))


def decode_img(blob):
    if isinstance(blob, dict):
        blob = blob.get("bytes", blob)
    return Image.open(io.BytesIO(blob)).convert("RGB").resize((128, 128))

print("decoding LIBERO frames...")
imgs = [decode_img(df.iloc[i]["observation.images.image"]) for i in range(n_frames)]
task_ids = df["task_index"].values
print(f"got {len(imgs)} frames, tasks present: {sorted(set(task_ids.tolist()))}")


def make_contact_sheet(top_indices: list[int], values: list[float], task_ids: list[int],
                       title: str, save_path: Path, n_cols: int = 6):
    """Horizontal strip of frames with activation + task above each."""
    import matplotlib.pyplot as plt
    n = len(top_indices)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.4), constrained_layout=True)
    if n == 1: axes = [axes]
    for ax, idx, v, tid in zip(axes, top_indices, values, task_ids):
        ax.imshow(imgs[idx])
        ax.set_title(f"act={v:.2f}\nframe={idx}", fontsize=7)
        ax.set_xlabel(f"task {tid}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=8)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---- Pick candidate features ----
# Three groups:
#   A. low-CV (semantic candidates) with reasonable fire rate
#   B. high-CV (flow-time-varying) with reasonable fire rate
#   C. highest peak activation (regardless)
fire_ok = (fire_rate > 0.05) & (fire_rate < 0.95)  # not always-on, not never-on
low_cv  = np.where(fire_ok & (med_cv < 0.2))[0]
high_cv = np.where(fire_ok & (med_cv > 1.0))[0]

# Sort each group by max peak activation
peak = per_frame.max(axis=0)
low_cv  = sorted(low_cv,  key=lambda f: -peak[f])[:10]
high_cv = sorted(high_cv, key=lambda f: -peak[f])[:10]

print(f"Candidate counts: low_cv={len(low_cv)}, high_cv={len(high_cv)}")

# ---- Generate contact sheets for top 6 in each group ----
N_TOP = 6
selected = {"low_cv": low_cv[:N_TOP], "high_cv": high_cv[:N_TOP]}
manifest = {}
for group, feats in selected.items():
    for f in feats:
        # Top frames for this feature
        top_frame_idx = np.argsort(-per_frame[:, f])[:6]
        vals = per_frame[top_frame_idx, f]
        tids = [int(task_ids[i]) for i in top_frame_idx]
        title = f"feat {f} ({group})  fire_rate={fire_rate[f]:.2f}  CV={med_cv[f]:.2f}"
        make_contact_sheet(list(top_frame_idx), vals.tolist(), tids,
                           title, OUT / f"feat_{f:05d}.png")
        manifest[int(f)] = {
            "group": group, "fire_rate": float(fire_rate[f]),
            "median_cv": float(med_cv[f]), "peak": float(peak[f]),
            "top_frames": [int(i) for i in top_frame_idx],
            "top_values": [float(v) for v in vals.tolist()],
            "top_task_ids": tids,
            "top_task_texts": [task_id_to_text.get(t, str(t))[:60] for t in tids],
        }

(OUT / "index.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"\nsaved {len(manifest)} feature sheets to {OUT}/")
print(f"index: {OUT/'index.json'}")
