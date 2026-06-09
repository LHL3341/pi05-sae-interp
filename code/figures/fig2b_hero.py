"""Figure 2(b): hero figure — physical vs semantic feature pair.

Top row: a high-CV feature whose top-activating frames concentrate on a single
LIBERO task and span a contiguous slice of one episode → "physical/dynamic"
motion-phase feature.

Bottom row: a low-CV feature whose top-activating frames span all 3 tasks →
"semantic" task-invariant feature.

Side-by-side support for the proposal's Aim 3 hypothesis: the SAE dictionary
contains both phases of decoupled signal in the continuous-time vector field.
"""
from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# Hero pair — picked from the v2 feature browser
HIGH_CV_FEAT = 2319
LOW_CV_FEAT  = 6951
N_COLS = 5

# ---- replicate stratified sampling so frame indices align ----
parquets = sorted((ROOT / "datasets/libero_10_image/data/chunk-000").glob("file-*.parquet"))
dfs = [pq.read_table(p).to_pandas() for p in parquets]
full = pd.concat(dfs, ignore_index=True)
n_frames = 480
eps = sorted(full["episode_index"].unique().tolist())
per_ep = max(1, n_frames // len(eps))
chunks = []
for ep in eps:
    sub = full[full["episode_index"] == ep]
    n_ep = min(per_ep, len(sub))
    if n_ep == 0: continue
    step = max(1, len(sub) // n_ep)
    chunks.append(sub.iloc[::step].iloc[:n_ep])
df = pd.concat(chunks, ignore_index=True).iloc[:n_frames].reset_index(drop=True)


def decode(blob, sz=160):
    if isinstance(blob, dict):
        blob = blob.get("bytes", blob)
    return Image.open(io.BytesIO(blob)).convert("RGB").resize((sz, sz))


index = json.loads((OUT / "feature_browser_v2/index.json").read_text())
high = index[str(HIGH_CV_FEAT)]
low  = index[str(LOW_CV_FEAT)]


def panel(ax_row, info, title, color):
    top = info["top_frames"][:N_COLS]
    vals = info["top_values"][:N_COLS]
    tids = info["top_task_ids"][:N_COLS]
    for ax, idx, v, tid in zip(ax_row, top, vals, tids):
        img = decode(df.iloc[idx]["observation.images.image"])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"a={v:.1f}", fontsize=7, color=color, pad=1)
        ax.set_xlabel(f"frame {idx} · task {tid}", fontsize=6)
    ax_row[0].set_ylabel(title, fontsize=7.5, color=color, rotation=90, labelpad=2)


fig, axes = plt.subplots(2, N_COLS, figsize=(1.5 * N_COLS, 3.4), constrained_layout=True)
panel(axes[0],
      high,
      f"feat {HIGH_CV_FEAT} (high-CV)\n"
      f"CV={high['median_cv']:.2f}  fire={high['fire_rate']:.2f}\n"
      f"task-specific motion phase",
      color="C3")
panel(axes[1],
      low,
      f"feat {LOW_CV_FEAT} (low-CV)\n"
      f"CV={low['median_cv']:.2f}  fire={low['fire_rate']:.2f}\n"
      f"cross-task semantic",
      color="C0")

fig.savefig(OUT / "fig2b_hero.pdf", dpi=300)
fig.savefig(OUT / "fig2b_hero.png", dpi=300)
print(f"saved {OUT/'fig2b_hero.pdf'}")
print()
print(f"high-CV feat {HIGH_CV_FEAT}: tasks of top 5 = {high['top_task_ids'][:N_COLS]}")
print(f"low-CV  feat {LOW_CV_FEAT}: tasks of top 5 = {low['top_task_ids'][:N_COLS]}")
