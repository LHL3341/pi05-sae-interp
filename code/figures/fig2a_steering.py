"""Figure 2(a): steering response — trained vs random π0.5 + LIBERO obs.

Plots ‖Δaction‖ as a function of steering coefficient α along (i) SAE-feature
directions for the trained model (saturated) and (ii) the same SAE direction
for an untrained π0.5 of identical architecture (linear). Optionally overlay
random Gaussian directions on the trained model as a within-subject control.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

FEATURES = [7676, 2036, 1102]
ALPHAS = [-3.0, -1.0, 0.0, 1.0, 3.0]


def deltas_for(run_dir: Path):
    rows = []
    for f in FEATURES:
        base = np.load(run_dir / f"feat{f}_alpha+0.00.npy")
        norms = []
        for a in ALPHAS:
            d = np.load(run_dir / f"feat{f}_alpha{a:+.2f}.npy")
            norms.append(np.linalg.norm(d - base))
        rows.append(norms)
    return np.array(rows)  # [n_feat, n_alpha]


trained_sae = deltas_for(RUNS / "steer_real_v2")
trained_rand = deltas_for(RUNS / "steer_real_random_dir")
untrained_sae = deltas_for(RUNS / "steer_random_libero")  # only +alpha valid (NaN for -alpha)

fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)

# Plot mean ± std across the 3 features, separate panel by condition.
def line(ax, alphas, data, label, **kw):
    m, s = data.mean(axis=0), data.std(axis=0)
    ax.errorbar(alphas, m, yerr=s, label=label, capsize=2, lw=1.4, ms=4, **kw)

line(ax, ALPHAS, trained_sae,   "trained π0.5 · SAE dir",   marker="o", color="C0")
line(ax, ALPHAS, trained_rand,  "trained π0.5 · rand dir",  marker="s", color="C2", ls="--")
# untrained data has NaN for negative α — re-run was only +alphas in old setup
um = untrained_sae[:, [0,1,2,3,4]]
line(ax, ALPHAS, um, "untrained π0.5 · SAE dir", marker="^", color="C3", ls=":")

ax.axvline(0, color="0.7", lw=0.5)
ax.set_xlabel(r"steering coeff. $\alpha$")
ax.set_ylabel(r"$\|\Delta a\|_2$  (action units)")
ax.set_yscale("log")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Aim 4: feature steering response", fontsize=8)

fig.savefig(OUT / "fig2a_steering.pdf", dpi=300)
fig.savefig(OUT / "fig2a_steering.png", dpi=300)
print(f"saved {OUT/'fig2a_steering.pdf'}")
print()
print("=== summary stats (mean over 3 features) ===")
for label, arr in [("trained·SAE", trained_sae), ("trained·rand", trained_rand),
                   ("untrained·SAE", untrained_sae)]:
    print(f"  {label:14s} α∈{ALPHAS}: " +
          " ".join(f"{m:.2f}" for m in arr.mean(axis=0)))
