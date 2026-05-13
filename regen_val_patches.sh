#!/bin/bash
# val 패치 전체 재생성 (min_valid_frac=0.30)
set -euo pipefail

WEDDELL="/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea"
SRC="/home/pyuncb/src"
LABEL_DIR="$SRC/label_code/labels"
LOG="$SRC/logs/regen_val_patches.log"

mkdir -p "$SRC/logs"

declare -A SCENES=(
    ["LC08_L1GT_165110_20200302_20201016_02_T2"]="$WEDDELL/2020/03/20200302/LC08_L1GT_165110_20200302_20201016_02_T2"
    ["LC08_L1GT_171110_20200225_20201016_02_T2"]="$WEDDELL/2020/02/20200225/LC08_L1GT_171110_20200225_20201016_02_T2"
    ["LC08_L1GT_177110_20200219_20201016_02_T2"]="$WEDDELL/2020/02/20200219/LC08_L1GT_177110_20200219_20201016_02_T2"
    ["LC08_L1GT_181098_20200419_20201016_02_T2"]="$WEDDELL/2020/04/20200419/LC08_L1GT_181098_20200419_20201016_02_T2"
    ["LC08_L1GT_188114_20201114_20210315_02_T2"]="$WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2"
    ["LC08_L1GT_199110_20200128_20201016_02_T2"]="$WEDDELL/2020/01/20200128/LC08_L1GT_199110_20200128_20201016_02_T2"
)

echo "=== val 패치 재생성 시작: $(date) ===" | tee -a "$LOG"
echo "min_valid_frac=0.30  (label 파일 있는 6개 씬)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for scene_id in "${!SCENES[@]}"; do
    scene_dir="${SCENES[$scene_id]}"
    label_path="$LABEL_DIR/${scene_id}_labels.tif"

    echo "[$scene_id]" | tee -a "$LOG"
    conda run -n remote python "$SRC/label_code/scene_to_patches.py" \
        --scene_dir  "$scene_dir" \
        --label_path "$label_path" \
        --split val \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== 완료: $(date) ===" | tee -a "$LOG"
ls "$SRC/data/VALIDATION_ZARR/"*.zarr 2>/dev/null | wc -l | xargs echo "총 패치 수:" | tee -a "$LOG"
