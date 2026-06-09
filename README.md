# Interpretability of π₀.₅ via Sparse Autoencoders

A class research project investigating how sparse autoencoders (SAEs) decompose the vision-language-action interface in Physical Intelligence's π₀.₅ policy, with experiments on the LIBERO-10 benchmark.

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
```

## 未上传的大文件（复现说明）

由于 GitHub 仓库大小限制，以下文件已通过 `.gitignore` 排除，但**首次运行时会自动下载**，无需手动准备：

### 1. 模型权重（自动下载）
- **π₀.₅ 权重**: `code/checkpoints/pi05_libero_base/` (~14.5 GB)
  - 来源：Hugging Face `lerobot/pi05_libero_base`
  - 首次运行 `extract_real.py` 时通过 `transformers` 自动下载到本地
  - 包含：`model.safetensors.index.json` + 多个 `.safetensors` 分片 + `config.json`

### 2. 数据集（自动下载）
- **LIBERO-10 图像数据**: `code/datasets/lerobot___libero_10_image/` (~189 MB)
  - 来源：Hugging Face `lerobot/libero_10_image`
  - 首次运行 `extract_real.py` 时通过 `datasets` 库自动下载
  - 包含：10 个任务变体的 parquet 文件，每个任务 ~4-6 个 episode，共 843 帧

### 3. 中间产物（需重新生成）
这些文件是运行实验管道生成的，**不会自动下载**，需按下方流程重新跑：

- **激活值缓存**: `code/activations/real_pi05_libero/` (~6.4 GB)
  - 内容：480 帧 × 10 个 flow step × 1024 维的 `pre_action_proj.npy` + `meta.json`
  - 生成方式：运行步骤 1（`extract_real.py`）

- **训练好的 SAE**: `code/runs/sae_real/` (~204 MB)
  - 内容：`sae.pt` (8192 维字典) + `train_log.json`
  - 生成方式：运行步骤 2（`sae.py`）

- **分析结果**: `code/runs/analyze_real/` + `code/runs/steer_real/` (~10 MB)
  - 内容：特征 CV 统计、激活热力图、steering manifest
  - 生成方式：运行步骤 3-4（`analyze_features.py` + `steer.py`）

### 4. OpenPI 源码（需手动克隆）
- **OpenPI 仓库**: `code/openpi/` (~12 GB)
  - 来源：https://github.com/Physical-Intelligence/openpi
  - 用途：提供 `PI05Policy` 的完整模型定义（本项目已在 `extract_real.py` 中做了 JAX-free shim，实际只需很小部分）
  - 克隆方式：`cd code && git clone https://github.com/Physical-Intelligence/openpi.git`
  - **可选**：如果只想跑推理，shim 已足够，不强制克隆

---

## 完整复现流程

### 环境准备
```bash
# Python 3.10, CUDA 12.4, PyTorch 2.8
pip install torch==2.8.0 transformers==4.53.2 datasets pillow numpy matplotlib scipy tqdm

# 应用 transformers 补丁（匹配 lerobot 的 PI05Config 字段）
cd code && bash apply_patch.sh
```

### 运行管道（首次会自动下载 14.7 GB）
```bash
cd code

# 步骤 1: 提取激活值（首次运行会下载 π₀.₅ 权重 + LIBERO 数据集）
python extract_real.py --n-frames 480 --out activations/real_pi05_libero
# 预计耗时：权重下载 5-15 分钟（取决于网速），推理 10-20 分钟（GPU）

# 步骤 2: 训练 SAE
python sae.py \
  --act-dir activations/real_pi05_libero \
  --layer pre_action_proj \
  --out runs/sae_real \
  --epochs 30
# 预计耗时：30 epochs × 1-2 分钟/epoch = 30-60 分钟（单 GPU）

# 步骤 3: 分析特征
python analyze_features.py \
  --act-dir activations/real_pi05_libero \
  --sae-ckpt runs/sae_real/sae.pt \
  --out runs/analyze_real
# 生成：CV 统计、激活热力图、top-activating 帧索引

# 步骤 4: Steering 实验
python steer.py \
  --sae-ckpt runs/sae_real/sae.pt \
  --features 1102 2036 7676 \
  --alphas -3 -1 0 1 3 \
  --out runs/steer_real
# 对比：SAE 方向 vs. 随机方向 vs. 未训练模型
```

### 快速验证（跳过下载，使用随机权重）
```bash
# 用随机权重测试管道连通性（2 分钟内完成，不需要下载）
python extract_real.py --n-frames 32 --skip-weights --out activations/smoke_test
python sae.py --act-dir activations/smoke_test --layer pre_action_proj --out runs/smoke --epochs 5
```

### 常见问题
- **下载卡住**：设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像
- **CUDA OOM**：降低 `--n-frames` 或在 `extract_real.py` 中调小 batch size
- **权重路径错误**：确保 `code/checkpoints/` 有写权限，首次下载会在这里缓存

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
