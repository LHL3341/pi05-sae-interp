"""Real-π0.5 + real-LIBERO feature steering (Aim 4, real run).

Mirrors steer.py but loads lerobot's PI05Policy with real weights and uses real
LIBERO observations. The hook target is identical: ``model.action_out_proj``
input pre-pended with ``α · sae.W_dec[feature]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, "/root/mozbrain")

from extract_real import load_libero_episode  # noqa: E402
from sae import TopKSAE  # noqa: E402


def build_policy(ckpt_dir: Path, data_dir: Path, device: torch.device, skip_weights: bool = False):
    """Same loader path as extract_real.main — repeated here to keep deps minimal."""
    from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.types import FeatureType, PolicyFeature

    raw = json.loads((ckpt_dir / "config.json").read_text())
    image_features = {
        k: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(v["shape"]))
        for k, v in raw["input_features"].items() if v["type"] == "VISUAL"
    }
    state_feature = next((PolicyFeature(type=FeatureType.STATE, shape=tuple(v["shape"]))
                          for v in raw["input_features"].values() if v["type"] == "STATE"), None)
    action_feature = next((PolicyFeature(type=FeatureType.ACTION, shape=tuple(v["shape"]))
                           for v in raw["output_features"].values() if v["type"] == "ACTION"), None)
    cfg = PI05Config(
        chunk_size=raw["chunk_size"], n_action_steps=raw["n_action_steps"],
        max_state_dim=raw["max_state_dim"], max_action_dim=raw["max_action_dim"],
        resize_imgs_with_padding=tuple(raw["image_resolution"]),
        empty_cameras=raw.get("empty_cameras", 0),
        tokenizer_max_length=raw["tokenizer_max_length"], num_steps=raw["num_inference_steps"],
        paligemma_variant=raw["paligemma_variant"], action_expert_variant=raw["action_expert_variant"],
        dtype=raw["dtype"],
    )
    cfg.input_features = {**image_features, "observation.state": state_feature}
    cfg.output_features = {"action": action_feature}
    cfg.tokenizer_path = str(THIS / "paligemma_tokenizer.model")
    cfg.tokenizer_type = "sentencepiece"

    raw_stats = json.loads((data_dir / "meta/stats.json").read_text())
    dataset_stats = {}
    for feat in list(cfg.input_features) + list(cfg.output_features):
        for c in (feat, feat.replace(".images.image2", ".images.wrist_image")):
            if c in raw_stats:
                dataset_stats[feat] = {k: torch.tensor(v) for k, v in raw_stats[c].items()
                                       if k in ("min", "max", "mean", "std")}
                break

    policy = PI05Policy(cfg, dataset_stats=dataset_stats)
    if not skip_weights:
        policy._init_from_pretrained(str(ckpt_dir / "model.safetensors"))
    policy.to(device)
    policy.eval()
    return policy


class FeatureSteer:
    def __init__(self, model, decoder_dir: torch.Tensor, alpha: float):
        self.dir = decoder_dir.to(next(model.parameters()).device)
        self.alpha = alpha
        self.model = model
        self.handle = None

    def __enter__(self):
        def pre_hook(_mod, inp):
            return (inp[0] + self.alpha * self.dir,)
        self.handle = self.model.action_out_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=THIS / "checkpoints/pi05_libero_base")
    p.add_argument("--data-dir", type=Path, default=THIS / "datasets/libero_10_image")
    p.add_argument("--sae-ckpt", type=Path, required=True)
    p.add_argument("--features", type=int, nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[-3.0, -1.0, 0.0, 1.0, 3.0])
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-weights", action="store_true")
    p.add_argument("--random-direction", action="store_true",
                   help="Use random unit Gaussian direction (matched to SAE decoder norm) instead of SAE feature direction. Control for non-isotropic claim.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    sae_cfg = ckpt["config"]
    if sae_cfg["layer"] != "pre_action_proj":
        raise ValueError(f"steering only on pre_action_proj, got {sae_cfg['layer']}")
    sae = TopKSAE(sae_cfg["d_in"], sae_cfg["d_dict"], sae_cfg["k"], sae_cfg["k_aux"])
    sae.load_state_dict(ckpt["state_dict"])

    device = torch.device(args.device)
    policy = build_policy(args.ckpt_dir, args.data_dir, device, args.skip_weights)

    # Same N frames across all (feature, alpha) combos to isolate the perturbation.
    batches = load_libero_episode(
        args.data_dir / "data/chunk-000/file-000.parquet",
        args.data_dir / "meta/tasks.parquet",
        n_frames=args.n_rollouts)
    batches = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
               for b in batches]

    # Reorder so α=0 is computed first (baseline for all subsequent Δ).
    alphas = sorted(set(args.alphas), key=lambda a: (a != 0.0, a))

    g = torch.Generator().manual_seed(args.seed)
    results = []
    for f in args.features:
        if args.random_direction:
            # Match SAE decoder norm so magnitudes are comparable.
            ref = sae.W_dec[f]
            r = torch.randn(ref.numel(), generator=g)
            decoder_dir = r / r.norm() * ref.norm()
        else:
            decoder_dir = sae.W_dec[f]  # [D]
        baseline_traj = None
        for alpha in alphas:
            trajs = []
            for b in batches:
                if alpha == 0.0:
                    a = policy.select_action(b)
                else:
                    with FeatureSteer(policy.model, decoder_dir, alpha):
                        a = policy.select_action(b)
                trajs.append(a.cpu().float().numpy())
            stack = np.stack(trajs, axis=0)  # [R, 1, T, A]
            if alpha == 0.0:
                baseline_traj = stack
            delta = float(np.linalg.norm(stack - baseline_traj)) if baseline_traj is not None else float("nan")
            np.save(args.out / f"feat{f}_alpha{alpha:+.2f}.npy", stack)
            print(f"[steer] feat={f:5d} alpha={alpha:+.2f} ||Δaction||={delta:.3f}")
            results.append({"feature": int(f), "alpha": float(alpha),
                            "delta": delta, "shape": list(stack.shape)})
    (args.out / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
