#!/bin/bash
# ============================================================================
# 4-Stage Self-Training Pipeline for Landsat 8 Cloud/Shadow/Snow Detection
#
# Usage:
#     ./pipeline.sh <experiment_name> [input_mode] [gpu_ids]
#
# Input modes  (Zarr patches contain B1–B7 + B9, 8 spectral channels):
#   Preset:
#     swirndsi        - B2–B7 + NDSI                        (7 ch) [default]
#     swirndsi_pca3   - B2–B7 + NDSI + global PC1-3        (10 ch)  ← requires data/global_pca.npz
#     cirrus_ndsi     - B2–B7 + B9 + NDSI                  (8 ch)
#     cirrus_ndsindwi - B2–B7 + B9 + NDSI + NDWI           (9 ch)
#     all_cirrus      - B1–B7 + B9                         (8 ch)
#     allndsi         - B1–B7 + NDSI                       (8 ch)
#     swirndsindwi    - B2–B7 + NDSI + NDWI                (8 ch)
#     swirndwi        - B2–B7 + NDWI                       (7 ch)
#     all             - B1–B7                              (7 ch)
#     vnir            - B2–B5                              (4 ch)
#     rgb             - B2–B4                              (3 ch)
#   Custom (pass "custom" as mode and set --bands / --indices in train.py):
#     python train.py ... --inp_mode custom --bands B2 B3 B4 B5 B9 --indices NDSI
#
# Examples:
#     ./pipeline.sh weddell_exp1                           # swirndsi, GPU 0 1
#     ./pipeline.sh weddell_exp1 cirrus_ndsi "0 1"         # use Cirrus band
#     ./pipeline.sh weddell_exp1 allndsi "0"
#
# Prerequisite — create H5 patches once before training:
#     python make_landsat_data.py --mode train --path /path/to/landsat/scenes
# ============================================================================

set -e

EXP_NAME=${1:?"Usage: ./pipeline.sh <experiment_name> [input_mode] [gpu_ids]"}
INP_MODE=${2:-swirndsi}
GPU_IDS=${3:-"0 1"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_RUN="conda run -n remote"

echo "=============================================="
echo "Self-Training Pipeline: ${EXP_NAME}"
echo "Input Mode: ${INP_MODE}"
echo "GPUs: ${GPU_IDS}"
echo "=============================================="

# ------------------------------------------------------------------
# Stage 0: Train on QA_PIXEL (Fmask) noisy labels
#           Network: depth=5, filters=16 (smallest)
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 0: Training on QA_PIXEL labels..."
${CONDA_RUN} python train.py \
    -e ${EXP_NAME}_stage0 \
    -st 0 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 64 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 1: Generate pseudo-labels from stage 0 model,
#           then train a larger network (depth=5, filters=32)
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 1: Generating pseudo-labels from stage 0..."
${CONDA_RUN} python label_generation.py \
    -e ${EXP_NAME}_stage0 \
    -st 1 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 1: Training..."
${CONDA_RUN} python train.py \
    -e ${EXP_NAME}_stage1 \
    -st 1 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 64 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 2: depth=6, filters=24
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 2: Generating pseudo-labels from stage 1..."
${CONDA_RUN} python label_generation.py \
    -e ${EXP_NAME}_stage1 \
    -st 2 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 2: Training..."
${CONDA_RUN} python train.py \
    -e ${EXP_NAME}_stage2 \
    -st 2 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 64 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 3: Final network (depth=6, filters=32, largest)
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 3: Generating pseudo-labels from stage 2..."
${CONDA_RUN} python label_generation.py \
    -e ${EXP_NAME}_stage2 \
    -st 3 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 3: Training..."
${CONDA_RUN} python train.py \
    -e ${EXP_NAME}_stage3 \
    -st 3 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 64 \
    -gpu ${GPU_IDS}

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Best model: exp_data/${EXP_NAME}_stage3/model/model_best.pth"
echo "=============================================="
