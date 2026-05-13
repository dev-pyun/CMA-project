#!/bin/bash
# Generate validation zarr patches for all 6 validation scenes.
set -e

cd /home/pyuncb/src

LABEL_DIR=/home/pyuncb/src/label_code/labels
WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

declare -A SCENES
SCENES[LC08_L1GT_165110_20200302_20201016_02_T2]=$WEDDELL/2020/03/20200302/LC08_L1GT_165110_20200302_20201016_02_T2
SCENES[LC08_L1GT_171110_20200225_20201016_02_T2]=$WEDDELL/2020/02/20200225/LC08_L1GT_171110_20200225_20201016_02_T2
SCENES[LC08_L1GT_177110_20200219_20201016_02_T2]=$WEDDELL/2020/02/20200219/LC08_L1GT_177110_20200219_20201016_02_T2
SCENES[LC08_L1GT_181098_20200419_20201016_02_T2]=$WEDDELL/2020/04/20200419/LC08_L1GT_181098_20200419_20201016_02_T2
SCENES[LC08_L1GT_188114_20201114_20210315_02_T2]=$WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2
SCENES[LC08_L1GT_199110_20200128_20201016_02_T2]=$WEDDELL/2020/01/20200128/LC08_L1GT_199110_20200128_20201016_02_T2

for scene_id in "${!SCENES[@]}"; do
    scene_dir=${SCENES[$scene_id]}
    label_path=$LABEL_DIR/${scene_id}_labels.tif
    echo "=============================="
    echo "[START] $scene_id"
    echo "  scene_dir : $scene_dir"
    echo "  label_path: $label_path"
    python -u label_code/scene_to_patches.py \
        --scene_dir  "$scene_dir" \
        --label_path "$label_path" \
        --split val \
        --overwrite
    echo "[DONE] $scene_id"
done

echo "=============================="
echo "All validation patches generated."
