"""Feature dictionary analysis (Aim 3).

Loads a trained SAE + the activation cache, computes per-feature top-activating
samples (which (rollout, denoise_step, action_index) triples maximally activate
each feature), and dumps a per-feature report for qualitative inspection.

For continuous flow-matching, semantic features ought to be more stable across
denoise steps while physical/action features ought to vary with the timestep —
so we also report the activation variance across denoise steps as a coarse
"semanticness" score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae import TopKSAE


def load_meta_and_activations(act_dir: Path, layer: str):
    meta = json.loads((act_dir / "meta.json").read_text())
    arr = np.load(act_dir / f"{layer}.npy")  # [calls, B, T, D]
    return meta, arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act-dir", type=Path, required=True)
    p.add_argument("--sae-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_in"], cfg["d_dict"], cfg["k"], cfg["k_aux"]).to(args.device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()

    meta, arr = load_meta_and_activations(args.act_dir, cfg["layer"])
    calls, B, T, D = arr.shape
    flat = torch.from_numpy(arr.reshape(-1, D)).float()

    # encode in chunks
    all_codes = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], 8192):
            chunk = flat[i:i + 8192].to(args.device)
            z = sae.encode(chunk)
            codes = sae.topk(z, sae.k).cpu()
            all_codes.append(codes)
    codes = torch.cat(all_codes, dim=0)  # [N, d_dict]
    codes = codes.reshape(calls, B, T, -1)

    # per-feature top-activating positions
    flat_codes = codes.reshape(-1, codes.shape[-1])  # [N, d_dict]
    topv, topi = flat_codes.topk(args.top_n, dim=0)  # [top_n, d_dict]

    # variance across denoise steps (token-averaged, batch-averaged)
    # codes shape: [calls, B, T, d_dict]
    per_call_mean = codes.mean(dim=(1, 2))  # [calls, d_dict]
    var_across_calls = per_call_mean.var(dim=0)  # [d_dict]
    mean_across_calls = per_call_mean.mean(dim=0).clamp(min=1e-9)
    cv = (var_across_calls.sqrt() / mean_across_calls).numpy()

    report = {
        "config": cfg,
        "calls": calls,
        "B": B,
        "T": T,
        "d_dict": codes.shape[-1],
        "feature_stats": [],
    }
    for f in range(codes.shape[-1]):
        if topv[0, f].item() == 0:
            continue
        positions = []
        for n in range(args.top_n):
            idx = topi[n, f].item()
            call = idx // (B * T)
            rem = idx % (B * T)
            b = rem // T
            t = rem % T
            positions.append({"call": int(call), "batch": int(b), "token": int(t),
                              "value": float(topv[n, f].item())})
        report["feature_stats"].append({
            "feature": int(f),
            "max_activation": float(topv[0, f].item()),
            "fire_rate": float((flat_codes[:, f] > 0).float().mean().item()),
            "cv_across_calls": float(cv[f]),
            "top_positions": positions,
        })
    report["feature_stats"].sort(key=lambda d: d["max_activation"], reverse=True)
    (args.out / "feature_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {len(report['feature_stats'])}/{codes.shape[-1]} features fired; "
          f"saved to {args.out/'feature_report.json'}")


if __name__ == "__main__":
    main()
