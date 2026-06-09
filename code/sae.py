"""Top-K Sparse Autoencoder (Aim 2).

Trains a Top-K SAE on activations dumped by ``extract_activations.py``.
Top-K (Makhzani-style hard top-k on encoder pre-activations) is the
default choice in recent VLA SAE work (Swann et al., 2026) — sparsity is
controlled exactly by k rather than via an L1 weight that drifts.

Auxiliary losses:
  • reconstruction:  ‖x − decoder(topk(encoder(x)))‖²
  • aux-k (dead-feature reanimation, Gao et al.):  encourages dead latents
    to contribute by predicting the residual with the k_aux dead latents.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TopKSAE(nn.Module):
    """Top-K SAE with tied bias (encoder bias = -decoder bias projection),
    inspired by OpenAI's recipe in "Scaling and evaluating sparse autoencoders"."""
    def __init__(self, d_in: int, d_dict: int, k: int, k_aux: int = 256):
        super().__init__()
        self.d_in, self.d_dict, self.k, self.k_aux = d_in, d_dict, k, k_aux
        self.W_enc = nn.Parameter(torch.empty(d_in, d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Kaiming init for encoder, then tie decoder = encoderᵀ + small noise
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-6)
        # rolling stats for dead-feature detection
        self.register_buffer("activity", torch.zeros(d_dict, dtype=torch.long))
        self.register_buffer("steps_since_active", torch.zeros(d_dict, dtype=torch.long))

    def encode(self, x):
        # x: [N, d_in]
        z = (x - self.b_dec) @ self.W_enc + self.b_enc  # [N, d_dict]
        return z

    def topk(self, z, k):
        topk_vals, topk_idx = z.topk(k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, topk_idx, topk_vals)
        return F.relu(out)  # ReLU-Top-K is the standard variant

    def decode(self, codes):
        return codes @ self.W_dec + self.b_dec

    def forward(self, x, *, dead_thresh: int = 1_000):
        z = self.encode(x)
        codes = self.topk(z, self.k)
        x_hat = self.decode(codes)
        recon = F.mse_loss(x_hat, x)

        # auxiliary "k_aux dead feature" loss
        with torch.no_grad():
            active = (codes.abs().sum(dim=0) > 0)
            self.activity += active.long()
            self.steps_since_active += 1
            self.steps_since_active[active] = 0
            dead_mask = self.steps_since_active >= dead_thresh

        if dead_mask.any() and self.training:
            z_dead = z.masked_fill(~dead_mask, float("-inf"))
            k_eff = min(self.k_aux, int(dead_mask.sum().item()))
            if k_eff > 0:
                topk_vals, topk_idx = z_dead.topk(k_eff, dim=-1)
                aux_codes = torch.zeros_like(z)
                aux_codes.scatter_(-1, topk_idx, F.relu(topk_vals))
                aux_recon = self.decode(aux_codes)
                residual = x - x_hat.detach()
                aux_loss = F.mse_loss(aux_recon, residual)
            else:
                aux_loss = torch.zeros((), device=x.device)
        else:
            aux_loss = torch.zeros((), device=x.device)

        loss = recon + 1.0 / 32.0 * aux_loss
        # frac of dead latents (rolling)
        return {
            "loss": loss,
            "recon": recon.detach(),
            "aux": aux_loss.detach(),
            "x_hat": x_hat.detach(),
            "codes": codes.detach(),
            "frac_dead": dead_mask.float().mean().detach(),
        }


def load_activations(path: Path, layer: str) -> torch.Tensor:
    arr = np.load(path / f"{layer}.npy")  # [calls, B, T, D]
    arr = arr.reshape(-1, arr.shape[-1])  # flatten over calls × batch × tokens
    return torch.from_numpy(arr).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act-dir", type=Path, required=True)
    p.add_argument("--layer", default="pre_action_proj",
                   choices=["pre_action_proj", "expert_final", "vlm_final"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expansion", type=int, default=8,
                   help="d_dict = expansion * d_in (typical SAE values: 4-32)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--k-aux", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    X = load_activations(args.act_dir, args.layer)
    print(f"[data] {args.layer}: {tuple(X.shape)} d_in={X.shape[1]}")
    d_in = X.shape[1]
    d_dict = args.expansion * d_in
    sae = TopKSAE(d_in, d_dict, args.k, args.k_aux).to(args.device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"[sae] d_in={d_in} d_dict={d_dict} k={args.k} k_aux={args.k_aux}")
    log = []
    for ep in range(args.epochs):
        t0 = time.time()
        for (xb,) in loader:
            xb = xb.to(args.device)
            out = sae(xb)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            # decoder unit-norm constraint (standard SAE practice)
            with torch.no_grad():
                sae.W_dec.div_(sae.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6))
            opt.step()
        # epoch metrics
        with torch.no_grad():
            xb = X[: args.batch_size].to(args.device)
            out = sae(xb)
            l0 = (out["codes"] != 0).float().sum(dim=-1).mean().item()
            r2 = 1.0 - F.mse_loss(out["x_hat"], xb).item() / xb.var().item()
        msg = (f"[ep{ep:02d}] loss={out['loss'].item():.4f} recon={out['recon'].item():.4f} "
               f"aux={out['aux'].item():.4f} L0={l0:.1f} R²={r2:.3f} "
               f"dead={out['frac_dead'].item():.3f} dt={time.time()-t0:.1f}s")
        print(msg)
        log.append({"epoch": ep, "recon": out["recon"].item(), "aux": out["aux"].item(),
                    "L0": l0, "R2": r2, "dead": out["frac_dead"].item()})
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_in": d_in, "d_dict": d_dict, "k": args.k, "k_aux": args.k_aux,
                   "expansion": args.expansion, "layer": args.layer},
    }, args.out / "sae.pt")
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] saved {args.out/'sae.pt'}")


if __name__ == "__main__":
    main()
