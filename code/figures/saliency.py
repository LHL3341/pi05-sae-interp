"""Image-space saliency for SAE features (overlay heatmap on LIBERO frames).

For a given (frame, feature, denoise-step) we compute the input-space gradient
of the SAE feature activation w.r.t.\ the camera image and visualise its
magnitude on top of the original frame. This answers "what part of the
image makes feature $f$ fire", which the prior CV/heatmap analyses do not.

Pipeline per (feature, frame):
  1. Run the policy's prefix forward with ``observation.images.image`` set to a
     leaf tensor with ``requires_grad=True``.
  2. Run one denoise step (t=0.5, mid-flow) under ``torch.enable_grad()``.
  3. Take the feature activation at the SAE encoder output for action token 0.
  4. Backprop and reduce the gradient to a $(H, W)$ saliency map by
     ``|grad| \cdot |image|`` summed over channels.
  5. Smooth + colormap + overlay on the un-preprocessed image.
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/mozbrain")
from sae import TopKSAE  # noqa: E402
from extract_real import _resize_norm  # noqa: E402

OUT = ROOT / "figures"
DEVICE = "cuda"

# ---- features to visualize and their picked frames (from feature_browser_v2) ----
# Pick a few canonical features
FEAT_FRAMES = [
    (2319, [59, 62, 61, 58]),    # high-CV, task-0 transport phase
    (1098, [305, 306, 307, 308]), # high-CV, task-2 microwave/kitchen
    (6951, [58, 56, 449, 265]),   # low-CV, "open gripper near object" cross-task
    (2486, [156, 244, 196, 261]), # low-CV, cross-task
]


def build_policy_for_grad():
    """Build PI05Policy with weights, eval mode, *without* torch.compile."""
    from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.types import FeatureType, PolicyFeature

    ckpt_dir = ROOT / "checkpoints/pi05_libero_base"
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
    cfg.tokenizer_path = str(ROOT / "paligemma_tokenizer.model")
    cfg.tokenizer_type = "sentencepiece"

    raw_stats = json.loads((ROOT / "datasets/libero_10_image/meta/stats.json").read_text())
    dataset_stats = {}
    for feat in list(cfg.input_features) + list(cfg.output_features):
        for c in (feat, feat.replace(".images.image2", ".images.wrist_image")):
            if c in raw_stats:
                dataset_stats[feat] = {k: torch.tensor(v) for k, v in raw_stats[c].items()
                                       if k in ("min", "max", "mean", "std")}
                break
    policy = PI05Policy(cfg, dataset_stats=dataset_stats)
    policy._init_from_pretrained(str(ckpt_dir / "model.safetensors"))
    policy.to(DEVICE)
    policy.eval()
    return policy


def load_frames():
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
    return pd.concat(chunks, ignore_index=True).iloc[:n_frames].reset_index(drop=True)


def decode_pil(blob, sz=224):
    if isinstance(blob, dict): blob = blob.get("bytes", blob)
    return Image.open(io.BytesIO(blob)).convert("RGB").resize((sz, sz))


def make_batch(row, device):
    img_pil   = decode_pil(row["observation.images.image"], 224)
    wrist_pil = decode_pil(row["observation.images.wrist_image"], 224)
    img   = torch.from_numpy(np.array(img_pil)).float().permute(2,0,1).unsqueeze(0) / 255.0
    wrist = torch.from_numpy(np.array(wrist_pil)).float().permute(2,0,1).unsqueeze(0) / 255.0
    img.requires_grad_(True)  # ← we'll backprop into the third-person camera
    state = torch.from_numpy(np.array(row["observation.state"], dtype=np.float32)).unsqueeze(0)
    return {
        "observation.images.image":  img.to(device),
        "observation.images.image2": wrist.to(device),
        "observation.state":         state.to(device),
        "task":                      [row.get("task", "")],
    }, img  # also return the leaf to read .grad later


def feature_saliency(policy, sae, image_leaf, batch, feat: int, denoise_t: float = 0.5):
    """Compute |grad| of feature `feat` w.r.t. the third-person image."""
    # Hook the action_out_proj input on every forward call.
    captured = {}
    h = policy.model.action_out_proj.register_forward_hook(
        lambda _m, inp, _o: captured.setdefault("h", inp[0]))

    with torch.enable_grad():
        b = policy.normalize_inputs(batch)
        b = policy.normalize_targets(b)
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        lang_tokens, lang_masks = policy.prepare_language(b)

        # 1) prefix forward to populate KV cache.
        from lerobot.common.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_embs, prefix_pad, prefix_att = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_att_2d_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d)
        policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_kv = policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d, position_ids=prefix_pos,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)

        # 2) one denoise step at the requested t.
        bsize = state.shape[0]
        noise = torch.normal(0., 1., (bsize, policy.config.n_action_steps,
                                       policy.config.max_action_dim), device=DEVICE)
        t_vec = torch.tensor([denoise_t], dtype=torch.float32, device=DEVICE)
        _ = policy.model.denoise_step(state, prefix_pad, past_kv, noise, t_vec)
        h_exp = captured["h"]
        x = h_exp.reshape(-1, h_exp.shape[-1])
        z = sae.encode(x)
        loss = z[:, feat].max()
        loss.backward()
    h.remove()
    grad = image_leaf.grad.detach()  # [1, 3, 224, 224]
    sal = (grad.abs() * (image_leaf.detach() + 1.0)).sum(dim=1).squeeze(0)  # [224, 224]
    sal = sal.cpu().numpy()
    image_leaf.grad = None
    return sal, float(loss.item())


def overlay(img_arr_uint8: np.ndarray, sal: np.ndarray, smooth_sigma: float = 6.0):
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(sal, sigma=smooth_sigma)
    s = (s - s.min()) / max(s.max() - s.min(), 1e-6)
    cmap = plt.get_cmap("jet")
    heat = cmap(s)[..., :3]  # RGB
    blend = 0.55 * (img_arr_uint8 / 255.0) + 0.45 * heat
    return np.clip(blend, 0, 1)


def main():
    frames_df = load_frames()
    print("loading policy...")
    policy = build_policy_for_grad()
    ckpt = torch.load(ROOT / "runs/sae_real_v2/sae.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_in"], cfg["d_dict"], cfg["k"], cfg["k_aux"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"]); sae.eval()

    fig, axes = plt.subplots(len(FEAT_FRAMES), 4 + 1,
                             figsize=(1.7 * 5, 1.7 * len(FEAT_FRAMES)),
                             constrained_layout=True)

    for r, (feat, frame_idxs) in enumerate(FEAT_FRAMES):
        for c, fi in enumerate(frame_idxs):
            row = frames_df.iloc[fi]
            batch, leaf = make_batch(row, DEVICE)
            sal, val = feature_saliency(policy, sae, leaf, batch, feat)
            img_pil = decode_pil(row["observation.images.image"], 224)
            img_arr = np.array(img_pil)
            blend = overlay(img_arr, sal)
            ax = axes[r, c]
            ax.imshow(blend); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"f{fi}·t{int(row['task_index'])}\nact={val:.2f}", fontsize=7)
        # rightmost: just original first frame for reference
        ref = decode_pil(frames_df.iloc[frame_idxs[0]]["observation.images.image"], 224)
        axes[r, -1].imshow(np.array(ref))
        axes[r, -1].set_xticks([]); axes[r, -1].set_yticks([])
        axes[r, -1].set_title("(raw)", fontsize=7)
        axes[r, 0].set_ylabel(f"feat {feat}", fontsize=8, rotation=90)

    fig.suptitle("Image-space saliency of SAE features  (gradient × input)",
                 fontsize=9, y=1.01)
    fig.savefig(OUT / "fig_saliency.pdf", dpi=300)
    fig.savefig(OUT / "fig_saliency.png", dpi=300)
    print(f"saved {OUT/'fig_saliency.pdf'}")


if __name__ == "__main__":
    main()
