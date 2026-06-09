#!/usr/bin/env bash
# Wait for pi05_libero_base/model.safetensors then run the real pipeline.
# Aim 1 → 2 → 3 → 4 in one shot.
set -euo pipefail
cd "$(dirname "$0")"

CKPT=checkpoints/pi05_libero_base/model.safetensors
echo "[chain] waiting for $CKPT"
until [ -f "$CKPT" ]; do sleep 30; done
sleep 5
sz=$(stat -c %s "$CKPT")
echo "[chain] $CKPT ready ($((sz/1024/1024)) MB) — starting pipeline"

PY=/root/opt/venv/bin/python

echo "[chain] === Aim 1: extract real activations (256 frames) ==="
$PY extract_real.py --n-frames 256 --out activations/real_pi05_libero

echo "[chain] === Aim 2: train Top-K SAE ==="
$PY sae.py --act-dir activations/real_pi05_libero --layer pre_action_proj --out runs/sae_real --epochs 30 --batch-size 512

echo "[chain] === Aim 3: feature analysis ==="
$PY analyze_features.py --act-dir activations/real_pi05_libero --sae-ckpt runs/sae_real/sae.pt --out runs/analyze_real --top-n 8

echo "[chain] === Aim 4: top-3 features steering ==="
TOP3=$($PY -c "
import json
r = json.load(open('runs/analyze_real/feature_report.json'))
print(' '.join(str(s['feature']) for s in r['feature_stats'][:3]))
")
$PY steer_real.py --sae-ckpt runs/sae_real/sae.pt --features $TOP3 --alphas -3 -1 0 1 3 --out runs/steer_real --n-rollouts 2

echo "[chain] DONE all four aims on real π0.5 + real LIBERO."
