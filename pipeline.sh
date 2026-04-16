#!/bin/bash
# ============================================================================
# 4-Stage Self-Training Pipeline for Landsat 8 Cloud/Shadow/Snow Detection
#
# Usage:
#     ./pipeline.sh <experiment_name> [input_mode] [gpu_ids]
#
# Examples:
#     ./pipeline.sh weddell_exp1                      # default: swirndsi, GPU 0 1
#     ./pipeline.sh weddell_exp1 swirndsi "0 1"
#     ./pipeline.sh weddell_exp1 allndsi "0"
# ============================================================================

set -e

EXP_NAME=${1:?"Usage: ./pipeline.sh <experiment_name> [input_mode] [gpu_ids]"}
INP_MODE=${2:-swirndsi}
GPU_IDS=${3:-"0 1"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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
python train.py \
    -e ${EXP_NAME}_stage0 \
    -st 0 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 32 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 1: Generate pseudo-labels from stage 0 model,
#           then train a larger network (depth=5, filters=32)
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 1: Generating pseudo-labels from stage 0..."
python label_generation.py \
    -e ${EXP_NAME}_stage0 \
    -st 1 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 1: Training..."
python train.py \
    -e ${EXP_NAME}_stage1 \
    -st 1 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 32 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 2: depth=6, filters=24
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 2: Generating pseudo-labels from stage 1..."
python label_generation.py \
    -e ${EXP_NAME}_stage1 \
    -st 2 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 2: Training..."
python train.py \
    -e ${EXP_NAME}_stage2 \
    -st 2 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 32 \
    -gpu ${GPU_IDS}

# ------------------------------------------------------------------
# Stage 3: Final network (depth=6, filters=32, largest)
# ------------------------------------------------------------------
echo ""
echo ">>> STAGE 3: Generating pseudo-labels from stage 2..."
python label_generation.py \
    -e ${EXP_NAME}_stage2 \
    -st 3 \
    -ip ${INP_MODE} \
    -gpu ${GPU_IDS}

echo ">>> STAGE 3: Training..."
python train.py \
    -e ${EXP_NAME}_stage3 \
    -st 3 \
    -ip ${INP_MODE} \
    -lr 0.000001 \
    -ep 400 \
    -bs 32 \
    -gpu ${GPU_IDS}

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Best model: exp_data/${EXP_NAME}_stage3/model/model_best.pth"
echo "=============================================="
