#!/bin/bash
SCENES_ROOT=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea
LABELS_DIR=/home/pyuncb/src/label_code/labels

cd /home/pyuncb/src

for label_tif in "$LABELS_DIR"/*.tif; do
    scene_id=$(basename "$label_tif" _labels.tif)
    scene_dir=$(find "$SCENES_ROOT" -type d -name "$scene_id" 2>/dev/null | head -1)
    if [ -z "$scene_dir" ]; then
        echo "[SKIP] $scene_id — scene dir not found"
        continue
    fi
    echo "[$(date +%H:%M:%S)] Processing $scene_id ..."
    conda run -n remote python label_code/scene_to_patches.py \
        --scene_dir "$scene_dir" \
        --label_path "$label_tif" \
        --split val
done

echo "Done."
