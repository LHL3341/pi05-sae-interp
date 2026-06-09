"""Real-π0.5 + real-LIBERO activation extraction (Aim 1, real run).

Differences vs. ``extract_activations.py``:
  • Loads ``lerobot/pi05_libero_base`` weights via lerobot's ``PI05Policy``
    (state dict matches openpi PI0Pytorch — same module names).
  • Replays real LIBERO_10 trajectories (parquet) instead of synthetic obs.
  • Same hook target: ``policy.model.action_out_proj`` input.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

THIS = Path(__file__).resolve().parent
# lerobot is installed at /root/mozbrain/lerobot
sys.path.insert(0, "/root/mozbrain")


def _resize_norm(img_np: np.ndarray, target: int = 224) -> torch.Tensor:
    """LIBERO frames are 256×256 uint8 NHWC. PI05 expects 224×224, NCHW float32 in [0, 1]
    (multiplied by 2 - 1 inside ``prepare_images`` to map to [-1, 1])."""
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if t.shape[-1] != target or t.shape[-2] != target:
        t = F.interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # [3, H, W]


def load_libero_episode(parquet_path, tasks_path: Path,
                        n_frames: int, start: int = 0,
                        stratify_episodes: bool = False):
    """Yields a list of single-frame batches matching PI05Policy's expected schema.

    ``parquet_path`` may be a single Path or a list of Paths (concatenated).
    ``stratify_episodes=True`` evenly samples ``n_frames`` across all unique
    (file, episode_index) pairs to maximize task / scene diversity.
    """
    if isinstance(parquet_path, (list, tuple)):
        dfs = [pq.read_table(p).to_pandas().assign(_src=str(p)) for p in parquet_path]
        df = (
            __import__("pandas").concat(dfs, ignore_index=True)
        )
    else:
        df = pq.read_table(parquet_path).to_pandas()
    tasks_df = pq.read_table(tasks_path).to_pandas()
    # lerobot v3 uses pandas' "index_columns" feature so ``task`` ends up as the
    # row index and ``task_index`` as the only column. Swap → {int: text}.
    if "task_index" in tasks_df.columns:
        task_id_to_text = dict(zip(tasks_df["task_index"], tasks_df.index)) \
            if tasks_df.index.name == "task" \
            else dict(zip(tasks_df["task_index"], tasks_df["task"]))
    else:
        task_id_to_text = dict(zip(tasks_df.index, tasks_df["task"]))

    if stratify_episodes:
        eps = sorted(df["episode_index"].unique().tolist())
        per_ep = max(1, n_frames // len(eps))
        chunks = []
        for ep in eps:
            sub = df[df["episode_index"] == ep]
            # evenly spaced indices within the episode
            n_ep = min(per_ep, len(sub))
            if n_ep == 0: continue
            step = max(1, len(sub) // n_ep)
            chunks.append(sub.iloc[::step].iloc[:n_ep])
        rows = __import__("pandas").concat(chunks, ignore_index=True).iloc[:n_frames]
    else:
        rows = df.iloc[start:start + n_frames]
    batches = []
    for _, row in rows.iterrows():
        # Images are stored as PNG-encoded bytes in lerobot v3 image datasets.
        def _decode(blob):
            if isinstance(blob, dict):
                blob = blob.get("bytes", blob.get("image", blob))
            return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
        img = _decode(row["observation.images.image"])
        wrist = _decode(row["observation.images.wrist_image"])
        state = np.asarray(row["observation.state"], dtype=np.float32)
        task_text = task_id_to_text.get(int(row.get("task_index", 0)), "")
        batches.append({
            "observation.images.image": _resize_norm(img).unsqueeze(0),
            # PI05 config calls the second cam "image2"
            "observation.images.image2": _resize_norm(wrist).unsqueeze(0),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [task_text],
        })
    return batches


class ActivationRecorder:
    """Same hooks as extract_activations.py, but on lerobot's PI05FlowMatching."""
    def __init__(self, model):
        self.buf: dict[str, list[torch.Tensor]] = {}
        self._handles = []
        joint = model.paligemma_with_expert
        original_forward = joint.forward
        recorder = self

        def patched(*args, **kwargs):
            out, pkv = original_forward(*args, **kwargs)
            prefix_out, suffix_out = out
            if suffix_out is not None:
                recorder.buf.setdefault("expert_final", []).append(
                    suffix_out.detach().to(torch.float32).cpu())
            if prefix_out is not None:
                recorder.buf.setdefault("vlm_final", []).append(
                    prefix_out.detach().to(torch.float32).cpu())
            return out, pkv
        joint.forward = patched

        self._handles.append(model.action_out_proj.register_forward_hook(
            lambda _m, inp, _o: self.buf.setdefault("pre_action_proj", []).append(
                inp[0].detach().to(torch.float32).cpu())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=THIS / "checkpoints/pi05_libero_base")
    p.add_argument("--data-dir", type=Path, default=THIS / "datasets/libero_10_image")
    p.add_argument("--n-frames", type=int, default=64,
                   help="Number of frames to replay across episodes.")
    p.add_argument("--out", type=Path, default=THIS / "activations" / "real_pi05_libero")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-weights", action="store_true",
                   help="Don't load safetensors; use random init (smoke test path).")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] policy from {args.ckpt_dir}")
    from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    # The HF config.json has fields the local PI05Config doesn't accept (rename
    # drift between lerobot versions). Cherry-pick the architectural fields and
    # build the config manually instead of calling .from_pretrained.
    raw = json.loads((args.ckpt_dir / "config.json").read_text())
    image_features = {
        k: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(v["shape"]))
        for k, v in raw.get("input_features", {}).items() if v["type"] == "VISUAL"
    }
    state_feature = next((PolicyFeature(type=FeatureType.STATE, shape=tuple(v["shape"]))
                          for v in raw.get("input_features", {}).values()
                          if v["type"] == "STATE"), None)
    action_feature = next((PolicyFeature(type=FeatureType.ACTION, shape=tuple(v["shape"]))
                           for v in raw.get("output_features", {}).values()
                           if v["type"] == "ACTION"), None)

    cfg = PI05Config(
        chunk_size=raw["chunk_size"],
        n_action_steps=raw["n_action_steps"],
        max_state_dim=raw["max_state_dim"],
        max_action_dim=raw["max_action_dim"],
        resize_imgs_with_padding=tuple(raw["image_resolution"]),
        empty_cameras=raw.get("empty_cameras", 0),
        tokenizer_max_length=raw["tokenizer_max_length"],
        num_steps=raw["num_inference_steps"],
        paligemma_variant=raw["paligemma_variant"],
        action_expert_variant=raw["action_expert_variant"],
        dtype=raw["dtype"],
    )
    cfg.input_features = {**image_features}
    if state_feature: cfg.input_features["observation.state"] = state_feature
    cfg.output_features = {"action": action_feature} if action_feature else {}
    # PaliGemma SentencePiece tokenizer (256k vocab + 1024 image tokens). Pulled
    # from public big_vision GCS bucket if not present.
    tok_path = THIS / "paligemma_tokenizer.model"
    if not tok_path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model",
            tok_path)
    cfg.tokenizer_path = str(tok_path)
    cfg.tokenizer_type = "sentencepiece"

    # Normalize layer needs per-feature stats; the LIBERO dataset's stats.json
    # is the right source. Convert to torch tensors keyed by feature name.
    stats_path = args.data_dir / "meta/stats.json"
    raw_stats = json.loads(stats_path.read_text())
    dataset_stats: dict[str, dict[str, torch.Tensor]] = {}
    # PI05Config maps "image" -> "observation.images.image" already; we also need
    # to alias "wrist_image" -> "observation.images.image2" (the second cam slot
    # the model expects). VISUAL norm is IDENTITY → stats not strictly required
    # but providing them is harmless.
    for feat in list(cfg.input_features) + list(cfg.output_features):
        # try direct match, then known aliases
        candidates = [feat,
                      feat.replace(".images.image2", ".images.wrist_image")]
        for c in candidates:
            if c in raw_stats:
                dataset_stats[feat] = {k: torch.tensor(v) for k, v in raw_stats[c].items()
                                       if k in ("min", "max", "mean", "std")}
                break

    policy = PI05Policy(cfg, dataset_stats=dataset_stats)
    if not args.skip_weights:
        policy._init_from_pretrained(str(args.ckpt_dir / "model.safetensors"))
    else:
        print("[load] --skip-weights set; using random init")
    policy.to(args.device)
    policy.eval()
    print(f"[load] policy loaded. params={sum(p.numel() for p in policy.parameters())/1e6:.1f}M")

    rec = ActivationRecorder(policy.model)

    print(f"[data] reading frames from {args.data_dir}")
    parquets = sorted((args.data_dir / "data/chunk-000").glob("file-*.parquet"))
    tasks_path = args.data_dir / "meta/tasks.parquet"
    batches = load_libero_episode(parquets, tasks_path, args.n_frames,
                                   stratify_episodes=getattr(args, "stratify", True))
    print(f"[data] got {len(batches)} frames")

    with torch.no_grad():
        for i, b in enumerate(batches):
            b = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            actions = policy.select_action(b)
            if i % 8 == 0:
                print(f"[step {i}] action_shape={tuple(actions.shape)} "
                      f"buf={ {k: len(v) for k, v in rec.buf.items()} }")

    meta = {"ckpt": str(args.ckpt_dir), "n_frames": args.n_frames,
            "shapes": {k: list(v[0].shape) for k, v in rec.buf.items()},
            "counts": {k: len(v) for k, v in rec.buf.items()}}
    for name, tensors in rec.buf.items():
        # vlm_final has variable seq_len if prompt token count differs; pad-stack.
        max_seq = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] < max_seq:
                pad = torch.zeros(t.shape[0], max_seq - t.shape[1], t.shape[2])
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        arr = torch.stack(padded, dim=0).numpy()
        np.save(args.out / f"{name}.npy", arr)
        print(f"[save] {name}.npy shape={arr.shape}")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {args.out/'meta.json'}")


if __name__ == "__main__":
    main()
