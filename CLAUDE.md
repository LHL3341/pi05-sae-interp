# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A class research project: **"Interpretability of π0.5 via Sparse Autoencoders"** — train Top-K SAEs on activations at the π0.5 vision-language backbone ↔ flow-matching action expert interface, then qualitatively analyze and steer the learned features. The full proposal is at `proposal (1).pdf`. Team: 林泓霖 (system integration / activation extraction), 温子辰 (SAE implementation), 高鑫 (interpretability analysis), 李馨月 (steering / functional assessment).

**Paper length: 4 pages** (overrides the NeurIPS template's default 9-page main body — apply this when planning, writing, or trimming).

## Repository layout

```
school/                       ← project root (this directory)
├── .claude/skills/           ← paper-* skills; MUST live here, not under paper/
├── paper/                    ← LaTeX sources (NeurIPS 2026 template)
│   ├── main.tex              ← title already set to the project's title
│   ├── build.sh              ← compiles → ../<project-dir-name>.pdf (i.e. school.pdf)
│   ├── sections/             ← 0_abstract..5_conclusion + A_appendix
│   ├── figures/, tables/, refs.bib
│   └── neurips_2026.sty, mybst.bst
└── proposal (1).pdf          ← the original project proposal
```

The template ships `.claude/` inside `paper/`, but Claude Code only loads skills from the **project root**. The directory has already been moved to `school/.claude/`; do not move it back into `paper/`.

### Experiment code (`code/`)

End-to-end Aim 1→4 pipeline runs against a **random-weight π0.5** (real architecture, untrained — used to validate plumbing). Run via the system venv `/root/opt/venv/bin/python` (Py 3.10 + torch 2.8 + CUDA 12.4). The full openpi `Pi0Config`/JAX path is **not** imported; we ship a JAX-free shim in `extract_activations.py` and a torch-only `image_tools` shim — only `transformers` (4.53.2, with the `transformers_replace` patch applied) and `torch` are required.

```
code/
├── openpi/                        ← cloned Physical-Intelligence/openpi
├── apply_patch.sh                 ← copies transformers_replace into the active venv (idempotent)
├── extract_activations.py         ← Aim 1: forward hooks at action_out_proj input + joint-transformer outputs
├── sae.py                         ← Aim 2: Top-K SAE w/ aux-k dead-feature loss
├── analyze_features.py            ← Aim 3: per-feature top-activating positions + denoise-step CV
├── steer.py                       ← Aim 4: α·W_dec[f] feature steering on action_out_proj input
├── activations/{run_name}/        ← *.npy (pre_action_proj, expert_final, vlm_final) + meta.json
└── runs/{name}/                   ← SAE checkpoints, training logs, feature reports, steering manifests
```

**Hook target.** `pre_action_proj` (input to `action_out_proj`, shape `[num_flow_steps, B, action_horizon, 1024]`) is the proposal's "VL backbone ↔ flow-matching expert interface": it's exactly the hidden state that becomes the velocity field via a single linear layer. SAE/steering default to this layer. `expert_final` and `vlm_final` are also captured for completeness.

**Pipeline check (one-liner sequence).** With venv activated:
```
PATH=/root/opt/venv/bin:$PATH bash code/apply_patch.sh           # once per venv
cd code && /root/opt/venv/bin/python extract_activations.py --n-rollouts 32 --num-flow-steps 10 --out activations/random_pi05
/root/opt/venv/bin/python sae.py --act-dir activations/random_pi05 --layer pre_action_proj --out runs/sae_pre_action_proj --epochs 10
/root/opt/venv/bin/python analyze_features.py --act-dir activations/random_pi05 --sae-ckpt runs/sae_pre_action_proj/sae.pt --out runs/analyze_pre_action_proj
/root/opt/venv/bin/python steer.py --sae-ckpt runs/sae_pre_action_proj/sae.pt --features 4090 5638 162 --alphas -3 -1 0 1 3 --out runs/steer_pre_action_proj
```

**Real-pipeline scripts.** `extract_real.py` runs the same hook scheme on **real π0.5 weights + real LIBERO frames** via the locally-installed lerobot package (`/root/mozbrain/lerobot`):
- Weights: HF `lerobot/pi05_libero_base` (14.5 GB safetensors, downloads to `code/checkpoints/pi05_libero_base/`).
- Data: HF `lerobot/libero_10_image` parquet (a single chunk, ~98 MB, ~843 frames covers 4 episodes; 10 task variants total).
- Tokenizer: `code/paligemma_tokenizer.model` (4 MB, public big_vision GCS — paligemma is a gated HF repo and that path is blocked).
- Stats: `meta/stats.json` from the LIBERO dataset is wired into the `Normalize` layers.
- Use `--skip-weights` to smoke-test the pipeline against random weights before the safetensors finishes downloading.
- LIBERO has only 2 cameras (`image`, `wrist_image`) but π0.5 expects 3; `empty_cameras=1` in the config fills the third slot with -1, matching how the model was trained.

**Sanity-check sequence (real weights):**
```
cd code
/root/opt/venv/bin/python extract_real.py --n-frames 256 --out activations/real_pi05_libero
/root/opt/venv/bin/python sae.py --act-dir activations/real_pi05_libero --layer pre_action_proj --out runs/sae_real --epochs 30
/root/opt/venv/bin/python analyze_features.py --act-dir activations/real_pi05_libero --sae-ckpt runs/sae_real/sae.pt --out runs/analyze_real
/root/opt/venv/bin/python steer.py --sae-ckpt runs/sae_real/sae.pt --features <ids…> --alphas -3 -1 0 1 3 --out runs/steer_real
```

**Things to know about LIBERO + lerobot.**
- The local lerobot `PI05Config` has field-name drift vs. the HF checkpoint's `config.json` (`num_inference_steps`→`num_steps`, no `time_sampling_*`/`min_period`/etc.). `extract_real.py` builds the config manually rather than calling `from_pretrained`.
- The HF checkpoint expects `observation.images.image` and `observation.images.image2`; the LIBERO_10 dataset names them `observation.images.image` and `observation.images.wrist_image`. The data loader renames `wrist_image → image2` at load time.
- `n_action_steps=10` for LIBERO π0.5 (vs. 50 in Aloha base), so `pre_action_proj` per call has shape `[1, 10, 1024]` (not 50).
- π0.5's discrete-state-input lives in the language prompt: `"Task: {task}, State: {digitized_state};\nAction: "` — handled inside `policy.prepare_language`.

## Building the paper

```bash
cd paper && bash build.sh    # → school.pdf at paper/school.pdf
```

`build.sh` runs `pdflatex → bibtex → pdflatex × 2` non-interactively and silently. On compile failure, read `paper/main.log` directly — `build.sh` only prints "✗ Compilation failed".

**`pdflatex` and `bibtex` are not installed** in this environment. Either install texlive, or run via apptainer (`apptainer exec /path/to/texlive.sif bash build.sh`). Don't claim a build succeeded without verifying the PDF exists.

## Workflow skills

Use these via `Skill` (or the user invokes `/<name>`):

| Skill | When |
|-------|------|
| `paper-plan` | Generate the section-by-section claims/evidence outline before drafting. Reads `NARRATIVE_REPORT.md`, `STORY.md`, etc. if present; otherwise asks the user. |
| `paper-write` | Draft LaTeX section by section from the plan. |
| `paper-figure` | Generate publication-quality figures/tables from experiment results. |
| `paper-illustration` | AI-generated architecture/method diagrams (Gemini + Claude refinement loop). |
| `paper-compile` | Compile and auto-fix LaTeX errors. |
| `research-paper-writing` | Paragraph-level polish, claim-support alignment, self-review. |

When the user asks for outline/draft/figure/compile work, prefer the matching skill over ad-hoc edits — the skills encode the expected workflow and venue conventions.

## Conventions worth knowing

- The paper follows the template's fixed 5-section structure (Intro, Related Work, Method, Experiments, Conclusion) plus an appendix. Scale section lengths down from the template's defaults to fit 4 pages.
- `build.sh` derives the output PDF name from the **parent directory's basename**, so the artifact is `school.pdf` here. Don't rename `main.pdf` manually.
- `refs.bib` is empty — populate it as citations are added; the proposal lists three starting references (π0.5, OpenVLA, Swann et al. SAE).
