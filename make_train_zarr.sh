#!/bin/bash
# Train 씬 심볼릭 링크 생성 + Zarr 패치 생성
# 실행: bash make_train_zarr.sh

set -e

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea
TRAIN_DIR=/home/pyuncb/src/data/TRAIN
SRC_DIR=/home/pyuncb/src

mkdir -p "$TRAIN_DIR"

# 이미 존재하면 스킵, 없으면 심볼릭 링크 생성
link_scene() {
    local src="$1"
    local name
    name=$(basename "$src")
    local dst="$TRAIN_DIR/$name"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "  [skip] $name"
    else
        ln -s "$src" "$dst"
        echo "  [link] $name"
    fi
}

echo "=== Symlinking train scenes ==="

# ── Category 1 (과소탐지) ──────────────────────────────────────────────
link_scene "$WEDDELL/2020/11/20201114/LC08_L1GT_188115_20201114_20210315_02_T2"
link_scene "$WEDDELL/2020/11/20201114/LC08_L1GT_188116_20201114_20210315_02_T2"
link_scene "$WEDDELL/2020/10/20201017/LC08_L1GT_200112_20201017_20201105_02_T2"
link_scene "$WEDDELL/2020/10/20201031/LC08_L1GT_202113_20201031_20201106_02_T2"
link_scene "$WEDDELL/2020/10/20201015/LC08_L1GT_202114_20201015_20201105_02_T2"
link_scene "$WEDDELL/2020/11/20201110/LC08_L1GT_160110_20201110_20210316_02_T2"
link_scene "$WEDDELL/2020/12/20201220/LC08_L1GT_184111_20201220_20210310_02_T2"
link_scene "$WEDDELL/2020/01/20200101/LC08_L1GT_170110_20200101_20201016_02_T2"
link_scene "$WEDDELL/2020/02/20200207/LC08_L1GT_205111_20200207_20201016_02_T2"
link_scene "$WEDDELL/2020/03/20200309/LC08_L1GT_198110_20200309_20201016_02_T2"
link_scene "$WEDDELL/2020/03/20200304/LC08_L1GT_179111_20200304_20201016_02_T2"
link_scene "$WEDDELL/2020/03/20200326/LC08_L1GT_205098_20200326_20201016_02_T2"
link_scene "$WEDDELL/2020/02/20200220/LC08_L1GT_168110_20200220_20201016_02_T2"

# ── Category 2 (과대탐지) ─────────────────────────────────────────────
link_scene "$WEDDELL/2020/04/20200407/LC08_L1GT_209098_20200407_20201016_02_T2"

# ── Category 3 (그림자 과소탐지) ──────────────────────────────────────
link_scene "$WEDDELL/2020/11/20201130/LC08_L1GT_188113_20201130_20210316_02_T2"
link_scene "$WEDDELL/2020/10/20201018/LC08_L1GT_207112_20201018_20201105_02_T2"
link_scene "$WEDDELL/2020/01/20200117/LC08_L1GT_202114_20200117_20201016_02_T2"
link_scene "$WEDDELL/2020/02/20200220/LC08_L1GT_184112_20200220_20201016_02_T2"
link_scene "$WEDDELL/2020/12/20201201/LC08_L1GT_195115_20201201_20210312_02_T2"
link_scene "$WEDDELL/2020/01/20200110/LC08_L1GT_169109_20200110_20201016_02_T2"

# ── Category 4 (구름 정탐지) ──────────────────────────────────────────
link_scene "$WEDDELL/2020/02/20200222/LC08_L1GT_166109_20200222_20201016_02_T2"
link_scene "$WEDDELL/2020/12/20201207/LC08_L1GT_189114_20201207_20210313_02_T2"
link_scene "$WEDDELL/2020/01/20200128/LC08_L1GT_183111_20200128_20201016_02_T2"
link_scene "$WEDDELL/2020/10/20201031/LC08_L1GT_218104_20201031_20201106_02_T2"
link_scene "$WEDDELL/2020/10/20201020/LC08_L1TP_221097_20201020_20201106_02_T1"

# ── Category 5 (SKC 정탐지) ───────────────────────────────────────────
link_scene "$WEDDELL/2020/11/20201126/LC08_L1GT_160109_20201126_20210316_02_T2"
link_scene "$WEDDELL/2020/11/20201126/LC08_L1GT_160110_20201126_20210316_02_T2"
link_scene "$WEDDELL/2020/03/20200311/LC08_L1GT_212108_20200311_20201016_02_T2"
link_scene "$WEDDELL/2020/12/20201203/LC08_L1GT_209112_20201203_20210313_02_T2"

echo ""
echo "Linked scenes: $(ls "$TRAIN_DIR" | wc -l) scene(s) in $TRAIN_DIR"
echo ""

# ── 소스 경로 존재 여부 확인 ──────────────────────────────────────────
echo "=== Checking source paths ==="
MISSING=0
for scene in "$TRAIN_DIR"/*/; do
    if [ ! -e "$scene" ]; then
        echo "  [MISSING] $scene"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "경고: $MISSING 개 씬의 소스 경로를 찾을 수 없습니다."
    echo "경로를 확인하고 다시 실행하세요."
    exit 1
fi

echo "모든 소스 경로 확인 완료."
echo ""

# ── Zarr 패치 생성 ────────────────────────────────────────────────────
echo "=== Generating Zarr patches ==="
cd "$SRC_DIR"
conda run -n remote python -m utils.split_scene --mode train

echo ""
echo "=== Done ==="
echo "출력: $SRC_DIR/data/TRAIN_ZARR/"
