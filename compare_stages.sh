#!/bin/bash
# compare_stages.sh
# data/TRAIN/ 아래 모든 train 씬에 대해
# compare_scene.py를 stage 0~3으로 실행하고 FCI를 씬 폴더에 복사.
#
# 사용법:
#   ./compare_stages.sh <exp_base> [gpu] [inp_mode]
#
# 예시:
#   ./compare_stages.sh exp_swirndsi_pca3
#   ./compare_stages.sh exp_swirndsi_pca3 0 swirndsi_pca3

set -euo pipefail

EXP_BASE="${1:?Usage: $0 <exp_base> [gpu] [inp_mode]}"
GPU="${2:-0}"
INP_MODE="${3:-swirndsi}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/data/TRAIN"
FCI_BASE="/earth00_home/immj/Landsat/Image/Weddell_Sea/FCI"

echo "=============================================="
echo "compare_stages: ${EXP_BASE}"
echo "  GPU      : $GPU"
echo "  inp_mode : $INP_MODE"
echo "  씬 소스  : $TRAIN_DIR"
echo "=============================================="
echo ""

mapfile -t SCENES < <(ls "$TRAIN_DIR")

echo "처리할 씬 (${#SCENES[@]}개):"
for S in "${SCENES[@]}"; do echo "  $S"; done
echo ""

# ── 씬별 처리 ──────────────────────────────────────────────────────────
for SCENE_NAME in "${SCENES[@]}"; do
    SCENE_DIR="$TRAIN_DIR/$SCENE_NAME"
    # compare_scene.py가 vis_output/<scene_name>/ 폴더를 자동 생성
    SCENE_OUT_DIR="$SCRIPT_DIR/vis_output/$SCENE_NAME"

    echo "══════════════════════════════════════════════"
    echo "씬: $SCENE_NAME"

    # ── stage 0~3 순서로 compare_scene.py 실행 ──────────────────────
    for STAGE in 0 1 2 3; do
        EXP="${EXP_BASE}_stage${STAGE}"
        EXP_MODEL_DIR="$SCRIPT_DIR/exp_data/$EXP"

        if [ ! -d "$EXP_MODEL_DIR" ]; then
            echo "  [SKIP] stage${STAGE}: 실험 디렉토리 없음"
            continue
        fi

        echo "  [stage${STAGE}] $EXP"
        conda run -n remote python "$SCRIPT_DIR/compare_scene.py" \
            --scene_dir "$SCENE_DIR" \
            --exp       "$EXP" \
            --stage     "$STAGE" \
            --inp_mode  "$INP_MODE" \
            --gpu       $GPU \
            --out       "$SCRIPT_DIR/vis_output" \
        && echo "    → 저장 완료" \
        || echo "    → [ERROR] 실행 실패"
    done

    # ── FCI 복사 ──────────────────────────────────────────────────────
    DATE="$(echo "$SCENE_NAME" | cut -d_ -f4)"
    YEAR="${DATE:0:4}"
    YM="${DATE:0:6}"
    FCI_SRC="${FCI_BASE}/${YEAR}/${YM}_FCI/${SCENE_NAME}.jpg"
    FCI_DST="${SCENE_OUT_DIR}/${SCENE_NAME}_FCI.jpg"

    if [ -f "$FCI_SRC" ]; then
        cp "$FCI_SRC" "$FCI_DST"
        echo "  [FCI] 복사 완료 → $(basename "$FCI_DST")"
    else
        echo "  [FCI] 없음: $FCI_SRC"
    fi

    echo ""
done

echo "=============================================="
echo "완료"
echo "=============================================="
