# Interpretability of π₀.₅ via Sparse Autoencoders

A class research project investigating how sparse autoencoders (SAEs) decompose the vision-language-action interface in Physical Intelligence's π₀.₅ policy, with experiments on the LIBERO-10 benchmark.

## Team

- **林泓霖** (Honglin Lin): system integration, activation extraction  
- **温子辰** (Zichen Wen): SAE implementation  
- **高鑫** (Xin Gao): interpretability analysis  
- **李馨月** (Xinyue Li): steering and functional assessment

## Quick Summary

We hook the input to π₀.₅'s velocity projection layer (the VL backbone ↔ flow-matching expert boundary), train a Top-K SAE (k=32, d_dict=8192) on 480 LIBERO frames, and find:

- **Faithful reconstruction**: R²=0.987, ~46% active features  
- **Semantic/physical split**: 2,215 low-CV features (stable across denoise steps, cross-task semantic) vs 1,594 high-CV features (vary with flow time, task-specific motion)  
- **Saturation under steering**: Adding α·W_dec[f] to the trained model perturbs actions by ||Δa|| ≈ 0.5 regardless of α or feature choice, 25× below the untrained baseline—evidence that supervised flow-matching compresses the interface into a robust, near-flat manifold

## Repository Layout

```
school/
├── CLAUDE.md              # project context for Claude Code
├── paper/                 # 4-page NeurIPS 2026 template
│   ├── main.tex
│   ├── sections/          # 0_abstract..5_conclusion + A_appendix
│   ├── figures/           # fig1.pdf (steering+CV+saliency), fig2.pdf (heatmap)
│   ├── refs.bib
│   └── build.sh           # pdflatex → bibtex → pdflatex × 2
├── code/
│   ├── extract_real.py    # Aim 1: hook π₀.₅, capture activations from real LIBERO data
│   ├── sae.py             # Aim 2: Top-K SAE with aux-k dead-feature loss
│   ├── analyze_features.py # Aim 3: CV, top-activating frames, heatmap
│   ├── steer.py           # Aim 4: feature steering via forward pre-hooks
│   ├── figures/           # final figure scripts + PDFs
│   └── paligemma_tokenizer.model  # 4 MB SentencePiece (public big_vision GCS)
└── proposal (1).pdf       # original project proposal
```

**Not included** (see `.gitignore`): 14.5 GB π₀.₅ weights (`checkpoints/`), LIBERO parquet (`datasets/`), 6.4 GB activation dumps (`activations/`), 204 MB trained SAEs (`runs/`), 12 GB openpi clone.

## Running the Pipeline

**Requirements**: Python 3.10, torch 2.8, CUDA 12.4, transformers 4.53.2  
**System venv used in development**: `/root/opt/venv/bin/python`

```bash
# 1. Extract activations from real π₀.₅ + LIBERO-10
cd code
/root/opt/venv/bin/python extract_real.py --n-frames 480 --out activations/real_pi05_libero

# 2. Train Top-K SAE
/root/opt/venv/bin/python sae.py --act-dir activations/real_pi05_libero --layer pre_action_proj --out runs/sae_real --epochs 30

# 3. Analyze features (CV, heatmap, top activations)
/root/opt/venv/bin/python analyze_features.py --act-dir activations/real_pi05_libero --sae-ckpt runs/sae_real/sae.pt --out runs/analyze_real

# 4. Steering experiment
/root/opt/venv/bin/python steer.py --sae-ckpt runs/sae_real/sae.pt --features <ids...> --alphas -3 -1 0 1 3 --out runs/steer_real
```

**First run**: `extract_real.py` auto-downloads `lerobot/pi05_libero_base` (14.5 GB) and `lerobot/libero_10_image` parquet (~98 MB) via HF `datasets`.

## Compiling the Paper

```bash
cd paper && bash build.sh   # → school.pdf
```

Requires `pdflatex` and `bibtex`. On systems without TeX Live, use an Apptainer/Singularity image:

```bash
apptainer exec /path/to/texlive.sif bash build.sh
```

## Key Implementation Notes

- **Hook target**: `action_out_proj` input (`pre_action_proj`), shape `[num_flow_steps, B, action_horizon, 1024]`—the hidden state that becomes the velocity field via a single linear layer.  
- **JAX-free shims**: `extract_real.py` includes minimal torch-only replacements for `openpi.models.gemma` and `openpi.shared.image_tools` to avoid pulling in the full JAX dependency tree.  
- **transformers patch**: The local lerobot install uses a patched `transformers` module (`transformers_replace`) to match π₀.₅'s specific config requirements; `apply_patch.sh` copies it into the active venv (idempotent).  
- **LIBERO adaptation**: LIBERO has only 2 cameras but π₀.₅ expects 3; the config uses `empty_cameras=1` to fill the third slot with -1 (matching training).  
- **Gradient saliency**: Computed via a custom forward pass with a single denoise step and manual gradient accumulation (`∂z_j/∂image × image`), overlaid on RGB frames as a heatmap.

## Citation

Full project proposal at `proposal (1).pdf`. Code and paper produced for a 2026 class project; cite as:

```bibtex
@techreport{lin2026pi05sae,
  author = {Honglin Lin and Zichen Wen and Xin Gao and Xinyue Li},
  title = {Interpretability of $\pi_{0.5}$ via Sparse Autoencoders},
  year = {2026},
  note = {Class research project}
}
```

## License

MIT (code), CC BY 4.0 (paper).
