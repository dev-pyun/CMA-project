#!/bin/bash
# 씬 하나에 대해 Fmask + stage 1/2/3 모델 예측 이미지를 한 번에 생성
#
# 사용법:
#   bash vis_pipeline.sh --scene_dir <경로> --exp_base <실험명_접두사> [옵션]
#
# 필수 인자:
#   --scene_dir   원본 Landsat L1 씬 폴더
#   --exp_base    실험명 접두사 (예: swirndsindwi_trial1)
#                 → exp_data/{exp_base}_stage1, _stage2, _stage3 순서로 실행
#
# 옵션:
#   --stages      실행할 stage 목록 (기본: "1 2 3")
#   --label_path  수동 라벨 GeoTIFF (val 씬에만 사용, ground_truth.png 생성)
#   --gpu         GPU ID (기본: 0)
#   --out         결과 저장 디렉토리 (기본: vis_output/)
#
# 예시:
#   # GT 없는 train 씬
#   bash vis_pipeline.sh \
#       --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188115_20201114_20210315_02_T2 \
#       --exp_base swirndsindwi_trial1
#
#   # GT 있는 val 씬
#   bash vis_pipeline.sh \
#       --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
#       --exp_base swirndsindwi_trial1 \
#       --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif
#
#   # stage 2, 3만
#   bash vis_pipeline.sh \
#       --scene_dir ... --exp_base swirndsindwi_trial1 --stages "2 3"

set -e

# ── 기본값 ────────────────────────────────────────────────────────────
SCENE_DIR=""
EXP_BASE=""
STAGES="1 2 3"
LABEL_PATH=""
GPU=0
OUT="vis_output/"

# ── 인자 파싱 ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene_dir)   SCENE_DIR="$2";   shift 2 ;;
        --exp_base)    EXP_BASE="$2";    shift 2 ;;
        --stages)      STAGES="$2";      shift 2 ;;
        --label_path)  LABEL_PATH="$2";  shift 2 ;;
        --gpu)         GPU="$2";         shift 2 ;;
        --out)         OUT="$2";         shift 2 ;;
        *) echo "[ERROR] 알 수 없는 인자: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENE_DIR" || -z "$EXP_BASE" ]]; then
    echo "[ERROR] --scene_dir 와 --exp_base 는 필수입니다."
    exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="$(basename "$SCENE_DIR")"

echo "========================================"
echo "  씬: $SCENE_ID"
echo "  실험 접두사: $EXP_BASE"
echo "  stage: $STAGES"
echo "  출력: $OUT"
echo "========================================"

# ── label_path 옵션 조립 ─────────────────────────────────────────────
LABEL_OPT=""
if [[ -n "$LABEL_PATH" ]]; then
    LABEL_OPT="--label_path $LABEL_PATH"
fi

# ── stage 순서대로 실행 ───────────────────────────────────────────────
for STAGE in $STAGES; do
    EXP="${EXP_BASE}_stage${STAGE}"
    EXP_DIR="$SRC_DIR/exp_data/$EXP"

    if [[ ! -d "$EXP_DIR" ]]; then
        echo ""
        echo "[SKIP] $EXP — exp_data/$EXP 디렉토리 없음"
        continue
    fi

    CKPT="$EXP_DIR/model/model_best.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo ""
        echo "[SKIP] $EXP — model_best.pth 없음 (학습 미완료)"
        continue
    fi

    echo ""
    echo "[stage $STAGE] $EXP"
    conda run -n remote python "$SRC_DIR/compare_scene.py" \
        --scene_dir "$SCENE_DIR" \
        --exp       "$EXP" \
        --stage     "$STAGE" \
        --gpu       "$GPU" \
        --out       "$OUT" \
        $LABEL_OPT
done

echo ""
echo "========================================"
echo "완료. 결과: $OUT/$SCENE_ID/"
echo "========================================"
