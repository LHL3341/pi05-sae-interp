"""Feature steering / intervention (Aim 4).

For a chosen SAE feature ``f``, register a forward-pre-hook on
``action_out_proj`` that adds ``α · W_dec[f]`` to its input. Compare the
perturbed action trajectory to the unperturbed baseline. The proposal calls
this an exploratory probe of "nonlinear stability" of the continuous-time
vector field along specific dictionary directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS / "openpi" / "src"))

from extract_activations import Pi0Config, build_model, fake_observation  # noqa: E402
from sae import TopKSAE  # noqa: E402


class FeatureSteer:
    """Adds α · W_dec[feature] to the action_out_proj input."""
    def __init__(self, model, decoder_dir: torch.Tensor, alpha: float):
        self.model = model
        self.dir = decoder_dir.to(next(model.parameters()).device)  # [D]
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        def pre_hook(_mod, inp):
            x = inp[0]  # [B, action_horizon, D]
            return (x + self.alpha * self.dir,)
        self.handle = self.model.action_out_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-ckpt", type=Path, required=True)
    p.add_argument("--features", type=int, nargs="+", required=True,
                   help="feature indices to steer")
    p.add_argument("--alphas", type=float, nargs="+", default=[-3.0, -1.0, 0.0, 1.0, 3.0])
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    sae_cfg = ckpt["config"]
    if sae_cfg["layer"] != "pre_action_proj":
        raise ValueError(f"steering only implemented for pre_action_proj, got {sae_cfg['layer']}")
    sae = TopKSAE(sae_cfg["d_in"], sae_cfg["d_dict"], sae_cfg["k"], sae_cfg["k_aux"])
    sae.load_state_dict(ckpt["state_dict"])

    cfg = Pi0Config()
    device = torch.device(args.device)
    model = build_model(cfg, device)

    results = []
    # Use the SAME noise + obs across alpha sweeps so the only delta is steering.
    torch.manual_seed(0)
    obs_list = [fake_observation(cfg, device) for _ in range(args.n_rollouts)]
    noise_list = [torch.normal(0.0, 1.0,
                               (1, cfg.action_horizon, cfg.action_dim),
                               device=device) for _ in range(args.n_rollouts)]

    with torch.no_grad():
        for f in args.features:
            decoder_dir = sae.W_dec[f]  # [D]
            baselines = {}  # rollout -> baseline action tensor
            for alpha in args.alphas:
                trajectories = []
                for r in range(args.n_rollouts):
                    if alpha == 0.0:
                        actions = model.sample_actions(device, obs_list[r], noise=noise_list[r],
                                                        num_steps=args.num_flow_steps)
                    else:
                        with FeatureSteer(model, decoder_dir, alpha):
                            actions = model.sample_actions(device, obs_list[r],
                                                            noise=noise_list[r],
                                                            num_steps=args.num_flow_steps)
                    trajectories.append(actions.cpu().float().numpy())
                stack = np.stack(trajectories, axis=0)  # [R, 1, H, A]
                if alpha == 0.0:
                    baselines[f] = stack
                delta = float(np.linalg.norm(stack - baselines[f])) if f in baselines else float("nan")
                results.append({"feature": int(f), "alpha": float(alpha),
                                "trajectories_shape": list(stack.shape),
                                "delta_from_baseline": delta})
                np.save(args.out / f"feat{f}_alpha{alpha:+.2f}.npy", stack)
                print(f"[steer] feat={f} alpha={alpha:+.2f} ||Δ||={delta:.3f}")

    (args.out / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
