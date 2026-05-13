#!/bin/bash
# compare_stages.sh
# Gini 필터링된 패치에 대해 stage0~3 전부 비교 시각화
#
# 사용법:
#   ./compare_stages.sh <exp_base> [n_sample] [gpu] [label_dir] [min_gini]
#
# 예시:
#   ./compare_stages.sh swirndsi_trial2
#   ./compare_stages.sh swirndsi_trial2 8 0
#   ./compare_stages.sh swirndsi_trial2 5 "0 1"
#   ./compare_stages.sh swirndsi_trial2 5 0 label_code/labels 0.2

set -euo pipefail

EXP_BASE="${1:?Usage: $0 <exp_base> [n_sample] [gpu] [label_dir] [min_gini]}"
N_SAMPLE="${2:-5}"
GPU="${3:-0}"
LABEL_DIR="${4:-label_code/labels}"
MIN_GINI="${5:-0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_DIR="$SCRIPT_DIR/data/VALIDATION_ZARR"
OUT_DIR="$SCRIPT_DIR/vis_output/${EXP_BASE}_stages_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

# ── Gini 필터링 후 샘플링 (한 번만) ───────────────────────────────────
echo "  Gini 필터링 중 (min_gini=${MIN_GINI})..."
mapfile -t PATCHES < <(
    conda run -n remote python "$SCRIPT_DIR/visualize_comparison.py" \
        --patch     "$VAL_DIR" \
        --exp       "${EXP_BASE}_stage0" \
        --sample    "$N_SAMPLE" \
        --min_gini  "$MIN_GINI" \
        --list_only \
        2>/dev/null
)

if [ "${#PATCHES[@]}" -eq 0 ]; then
    echo "[ERROR] No zarr patches found in $VAL_DIR"
    exit 1
fi

echo "=== compare_stages: ${EXP_BASE} ==="
echo "  샘플 수    : ${#PATCHES[@]} (min_gini=${MIN_GINI})"
echo "  GPU        : $GPU"
echo "  출력 경로  : $OUT_DIR"
echo ""
echo "선택된 패치:"
for p in "${PATCHES[@]}"; do
    echo "  $(basename "$p")"
done
echo ""

# ── 각 패치에 대해 stage0~3 순서로 실행 ───────────────────────────────
for PATCH in "${PATCHES[@]}"; do
    PATCH_NAME="$(basename "$PATCH")"
    echo "──────────────────────────────────────────"
    echo "패치: $PATCH_NAME"

    for STAGE in 0 1 2 3; do
        EXP="${EXP_BASE}_stage${STAGE}"
        EXP_MODEL_DIR="$SCRIPT_DIR/exp_data/$EXP"

        if [ ! -d "$EXP_MODEL_DIR" ]; then
            echo "  [SKIP] stage${STAGE}: 실험 디렉토리 없음 ($EXP_MODEL_DIR)"
            continue
        fi

        echo "  [stage${STAGE}] $EXP"
        conda run -n remote python "$SCRIPT_DIR/visualize_comparison.py" \
            --patch     "$PATCH" \
            --exp       "$EXP" \
            --label_dir "$LABEL_DIR" \
            --gpu       $GPU \
            --out       "$OUT_DIR" \
        && echo "    → 저장 완료" \
        || echo "    → [ERROR] 실행 실패 (모델 없음 또는 오류)"
    done
done

echo ""
echo "=== 완료 ==="
echo "출력 디렉토리: $OUT_DIR"
ls "$OUT_DIR"
