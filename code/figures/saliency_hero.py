"""Compact saliency hero: 2 features × 5 frames, overlaid on LIBERO RGB.
Replaces the raw-frame hero (b) of Figure 1.
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/mozbrain")
from sae import TopKSAE  # noqa: E402
from figures.saliency import (  # noqa: E402
    build_policy_for_grad, load_frames, decode_pil, make_batch,
    feature_saliency, overlay,
)

OUT = ROOT / "figures"
DEVICE = "cuda"

# Same hero pair as fig 2(b)
HIGH_CV_FEAT = 2319
LOW_CV_FEAT  = 6951
N_COLS = 5

# Frames from feature_browser_v2 index
INDEX = json.loads((OUT / "feature_browser_v2/index.json").read_text())


def main():
    frames_df = load_frames()
    print("loading policy ...")
    policy = build_policy_for_grad()
    ckpt = torch.load(ROOT / "runs/sae_real_v2/sae.pt", map_location=DEVICE,
                      weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_in"], cfg["d_dict"], cfg["k"], cfg["k_aux"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"]); sae.eval()

    # fix seed so the denoise noise is reproducible
    torch.manual_seed(0)

    fig, axes = plt.subplots(2, N_COLS, figsize=(1.5 * N_COLS, 3.4),
                             constrained_layout=True)

    for r, (feat, color, label) in enumerate([
        (HIGH_CV_FEAT, "C3", f"feat {HIGH_CV_FEAT}\nhigh-CV={INDEX[str(HIGH_CV_FEAT)]['median_cv']:.2f}\ntask-specific\nmotion phase"),
        (LOW_CV_FEAT,  "C0", f"feat {LOW_CV_FEAT}\nlow-CV={INDEX[str(LOW_CV_FEAT)]['median_cv']:.2f}\ncross-task\nsemantic"),
    ]):
        info = INDEX[str(feat)]
        frame_idxs = info["top_frames"][:N_COLS]
        for c, fi in enumerate(frame_idxs):
            row = frames_df.iloc[fi]
            batch, leaf = make_batch(row, DEVICE)
            sal, val = feature_saliency(policy, sae, leaf, batch, feat)
            img_arr = np.array(decode_pil(row["observation.images.image"], 224))
            blend = overlay(img_arr, sal, smooth_sigma=8.0)
            ax = axes[r, c]
            ax.imshow(blend); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"a={val:.1f}", fontsize=7, color=color, pad=1)
            ax.set_xlabel(f"f{fi}·t{int(row['task_index'])}", fontsize=6)
            # also dump as a standalone PNG for fig2_combined to pick up
            (OUT / "saliency_tiles").mkdir(exist_ok=True)
            blend_uint8 = (np.clip(blend, 0, 1) * 255).astype("uint8")
            Image.fromarray(blend_uint8).save(
                OUT / f"saliency_tiles/feat{feat}_f{fi}.png")
            del leaf, batch
            torch.cuda.empty_cache()
        axes[r, 0].set_ylabel(label, fontsize=7, color=color, rotation=90, labelpad=2)

    fig.savefig(OUT / "fig2b_saliency.pdf", dpi=300)
    fig.savefig(OUT / "fig2b_saliency.png", dpi=300)
    print(f"saved {OUT/'fig2b_saliency.pdf'}")


if __name__ == "__main__":
    main()
