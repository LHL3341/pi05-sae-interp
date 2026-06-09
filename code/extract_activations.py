"""π0.5 activation extraction (Aim 1).

Hooks the π0.5 PyTorch model at the semantic ↔ flow-matching interface and
saves dense activations to disk for later SAE training.

Hook points:
  • pre_action_proj — input to ``action_out_proj``. Shape [B, action_horizon, D].
    This IS the interface the proposal targets: the action expert's final
    hidden state right before it becomes the velocity field.
  • expert_layer{i}  — action-expert residual stream after every joint
    transformer block (captured by wrapping ``PaliGemmaWithExpertModel.forward``).

Default config: ``debug_pi05`` (dummy weights, no download). Replace
``build_config`` to swap in real π0.5 weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS / "openpi" / "src"))

# Shadow openpi.shared.image_tools with a torch-only re-implementation so we
# don't have to install JAX (the original module imports jax at top-level).
import types as _types  # noqa: E402

def _install_image_tools_shim():
    if "openpi.shared.image_tools" in sys.modules:
        return
    import torch.nn.functional as F
    m = _types.ModuleType("openpi.shared.image_tools")

    def resize_with_pad_torch(images, height, width, mode="bilinear"):
        # mirrors openpi.shared.image_tools.resize_with_pad_torch (torch path)
        if images.shape[-1] <= 4:
            channels_last = True
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
        else:
            channels_last = False
            if images.dim() == 3:
                images = images.unsqueeze(0)
        b, c, ch, cw = images.shape
        ratio = max(cw / width, ch / height)
        rh, rw = int(ch / ratio), int(cw / ratio)
        resized = F.interpolate(images, size=(rh, rw), mode=mode,
                                align_corners=False if mode == "bilinear" else None)
        if images.dtype == torch.uint8:
            resized = resized.round().clamp(0, 255).to(torch.uint8)
        elif images.dtype == torch.float32:
            resized = resized.clamp(-1.0, 1.0)
        ph0, rem_h = divmod(height - rh, 2); ph1 = ph0 + rem_h
        pw0, rem_w = divmod(width - rw, 2); pw1 = pw0 + rem_w
        cv = 0 if images.dtype == torch.uint8 else -1.0
        padded = F.pad(resized, (pw0, pw1, ph0, ph1), value=cv)
        if channels_last:
            padded = padded.permute(0, 2, 3, 1)
        return padded

    m.resize_with_pad_torch = resize_with_pad_torch
    sys.modules["openpi.shared.image_tools"] = m
    if "openpi.shared" not in sys.modules or not hasattr(sys.modules["openpi.shared"], "__path__"):
        pkg = _types.ModuleType("openpi.shared")
        pkg.__path__ = [str(THIS / "openpi" / "src" / "openpi" / "shared")]
        sys.modules["openpi.shared"] = pkg
    sys.modules["openpi.shared"].image_tools = m

_install_image_tools_shim()

# JAX-free shim for openpi.models.gemma. The PyTorch path only consumes
# ``get_config(variant) -> Config(width, depth, mlp_dim, num_heads, num_kv_heads, head_dim)``.
import types as _types  # noqa: E402

def _install_gemma_shim():
    if "openpi.models.gemma" in sys.modules:
        return
    # Mark openpi/openpi.models as proper packages by giving them __path__.
    import importlib
    importlib.import_module("openpi")  # real package on sys.path
    importlib.import_module("openpi.models_pytorch")  # ensure real package init
    m = _types.ModuleType("openpi.models.gemma")

    @dataclasses.dataclass
    class Config:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict = dataclasses.field(default_factory=dict)

    _CFGS = {
        "dummy":     dict(width=64,   depth=4,  mlp_dim=128,    num_heads=8, num_kv_heads=1, head_dim=16),
        "gemma_300m":dict(width=1024, depth=18, mlp_dim=4096,   num_heads=8, num_kv_heads=1, head_dim=256),
        "gemma_2b":  dict(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
    }

    def get_config(variant: str) -> Config:
        if variant not in _CFGS:
            raise ValueError(f"Unknown variant: {variant}")
        return Config(**_CFGS[variant])

    m.Config = Config
    m.get_config = get_config
    m.PALIGEMMA_VOCAB_SIZE = 257_152
    sys.modules["openpi.models.gemma"] = m
    # The real openpi.models package may not exist on disk under our shim
    # (the directory does, but its __init__ tries to import from gemma…).
    # We register a synthetic package so submodule imports resolve.
    if "openpi.models" not in sys.modules or not hasattr(sys.modules["openpi.models"], "__path__"):
        pkg = _types.ModuleType("openpi.models")
        pkg.__path__ = [str(THIS / "openpi" / "src" / "openpi" / "models")]
        sys.modules["openpi.models"] = pkg
    sys.modules["openpi.models"].gemma = m

_install_gemma_shim()

# Stand-in for openpi.models.pi0_config.Pi0Config that drops the JAX dependency.
# Mirrors the fields PI0Pytorch reads via ``self.config``.
@dataclasses.dataclass(frozen=True)
class Pi0Config:
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int | None = None
    pi05: bool = True
    discrete_state_input: bool | None = None
    dtype: str = "float32"  # bfloat16 hits the "vision_tower must be float32" path
    pytorch_compile_mode: str | None = None  # disable torch.compile for hookability

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)


def build_model(cfg: Pi0Config, device: torch.device) -> torch.nn.Module:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    model = PI0Pytorch(cfg)
    model.eval()
    model.to(device)
    return model


IMAGE_RESOLUTION = (224, 224)  # matches openpi.models.model.IMAGE_RESOLUTION


def fake_observation(cfg: Pi0Config, device: torch.device, prompt_tokens: int = 16):
    """Synthetic observation matching the model's expected input spec."""
    @dataclasses.dataclass
    class Obs:
        images: dict
        image_masks: dict
        state: torch.Tensor
        tokenized_prompt: torch.Tensor
        tokenized_prompt_mask: torch.Tensor
        token_ar_mask: torch.Tensor
        token_loss_mask: torch.Tensor

    H, W = IMAGE_RESOLUTION
    B = 1
    img_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images = {k: torch.rand(B, 3, H, W, device=device) * 2 - 1 for k in img_keys}
    image_masks = {k: torch.ones(B, dtype=torch.bool, device=device) for k in img_keys}
    state = torch.zeros(B, cfg.action_dim, device=device)
    prompt = torch.randint(0, 257_152, (B, prompt_tokens), device=device, dtype=torch.long)
    prompt_pad = torch.zeros(B, cfg.max_token_len - prompt_tokens, dtype=torch.long, device=device)
    prompt = torch.cat([prompt, prompt_pad], dim=1)
    prompt_mask = torch.cat(
        [torch.ones(B, prompt_tokens, dtype=torch.bool, device=device),
         torch.zeros(B, cfg.max_token_len - prompt_tokens, dtype=torch.bool, device=device)],
        dim=1,
    )
    token_ar = torch.zeros(B, cfg.max_token_len, dtype=torch.bool, device=device)
    token_loss = torch.zeros(B, cfg.max_token_len, dtype=torch.bool, device=device)
    return Obs(images, image_masks, state, prompt, prompt_mask, token_ar, token_loss)


class ActivationRecorder:
    """Forward-hooks pre_action_proj input + per-layer action-expert residual."""
    def __init__(self, model):
        self.model = model
        self.buf: dict[str, list[torch.Tensor]] = {}
        self._handles = []
        self._wrap_joint_forward()
        self._handles.append(
            model.action_out_proj.register_forward_hook(self._mk_input_hook("pre_action_proj"))
        )

    def _mk_input_hook(self, name):
        def hook(_mod, inp, _out):
            x = inp[0].detach().to(torch.float32).cpu()
            self.buf.setdefault(name, []).append(x)
        return hook

    def _wrap_joint_forward(self):
        """``PaliGemmaWithExpertModel.forward`` rebinds ``inputs_embeds`` per layer
        instead of calling ``layer.forward``, so a register_forward_hook on
        individual blocks never fires. Wrap the joint forward and snapshot the
        final outputs (vlm prefix + action expert suffix) on every call —
        including denoise-step suffix-only calls."""
        joint = self.model.paligemma_with_expert
        original_forward = joint.forward
        recorder = self

        def patched(*args, **kwargs):
            out, pkv = original_forward(*args, **kwargs)
            prefix_out, suffix_out = out
            if suffix_out is not None:
                recorder.buf.setdefault("expert_final", []).append(
                    suffix_out.detach().to(torch.float32).cpu()
                )
            if prefix_out is not None:
                recorder.buf.setdefault("vlm_final", []).append(
                    prefix_out.detach().to(torch.float32).cpu()
                )
            return out, pkv

        joint.forward = patched

    def close(self):
        for h in self._handles:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=THIS / "activations" / "debug_pi05")
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--num-flow-steps", type=int, default=10,
                   help="Flow-matching solver steps; per-step expert states all logged.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Pi0Config()
    device = torch.device(args.device)
    print(f"[build] device={device} cfg={cfg}")
    t0 = time.time()
    model = build_model(cfg, device)
    print(f"[build] done in {time.time()-t0:.1f}s "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    rec = ActivationRecorder(model)

    with torch.no_grad():
        for r in range(args.n_rollouts):
            obs = fake_observation(cfg, device)
            actions = model.sample_actions(device, obs, num_steps=args.num_flow_steps)
            print(f"[rollout {r}] actions={tuple(actions.shape)} "
                  f"buf={ {k: len(v) for k, v in rec.buf.items()} }")

    # save
    meta = {
        "config": dataclasses.asdict(cfg),
        "n_rollouts": args.n_rollouts,
        "num_flow_steps": args.num_flow_steps,
        "shapes": {k: list(v[0].shape) for k, v in rec.buf.items()},
        "counts": {k: len(v) for k, v in rec.buf.items()},
    }
    for name, tensors in rec.buf.items():
        arr = torch.stack(tensors, dim=0).numpy()
        np.save(args.out / f"{name}.npy", arr)
        print(f"[save] {name}.npy shape={arr.shape} dtype={arr.dtype}")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] meta saved to {args.out/'meta.json'}")


if __name__ == "__main__":
    main()
