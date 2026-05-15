"""
GT 라벨 오프셋 진단 스크립트

zarr 패치에 저장된 label이 spectral과 같은 지리 위치를 가리키는지 검사.

사용법:
    python diagnose_label_offset.py \
        --patch data/VALIDATION_ZARR/LC08_L1GT_188114_..._PATCH5.zarr \
        --label_path label_code/labels/LC08_L1GT_188114_..._labels.tif \
        --scene_dir  /earth00_home/.../LC08_L1GT_188114_...
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import zarr
import rasterio
from rasterio import windows as rwin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_comparison import parse_patch_name, find_scene_dir, find_patch_coords
from utils.split_scene import find_band_file, find_qa_pixel_file

PATCH_SIZE = 256
STRIDE     = 256

LABEL_REMAP = {0: 255, 1: 0, 2: 0, 3: 2, 4: 1, 255: 255}

def remap(arr):
    out = np.full_like(arr, 255, dtype=np.uint8)
    for s, d in LABEL_REMAP.items():
        out[arr == s] = d
    return out

def load_label_patch(label_path, row, col):
    with rasterio.open(label_path) as src:
        raw = src.read(1).astype(np.uint8)
    return remap(raw[row:row+PATCH_SIZE, col:col+PATCH_SIZE])

def match_score(a, b):
    """255(ignore) 제외하고 일치율 반환."""
    mask = (a != 255) & (b != 255)
    if mask.sum() == 0:
        return float('nan')
    return float((a[mask] == b[mask]).mean())


def get_label_tif_info(label_path):
    with rasterio.open(label_path) as src:
        return src.height, src.width, src.transform, src.crs

def get_band_info(scene_dir):
    bf = find_band_file(str(scene_dir), 'B4')
    with rasterio.open(bf) as src:
        return src.height, src.width, src.transform, src.crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch',      required=True)
    ap.add_argument('--label_path', required=True)
    ap.add_argument('--scene_dir',  default=None)
    ap.add_argument('--label_dir',  default='label_code/labels')
    args = ap.parse_args()

    patch_path  = args.patch
    label_path  = Path(args.label_path)
    scene_id, patch_idx = parse_patch_name(patch_path)

    scene_dir = Path(args.scene_dir) if args.scene_dir else find_scene_dir(scene_id)

    print(f"\n=== 진단: {scene_id}  PATCH{patch_idx} ===\n")

    # ── 1. label TIF vs raw band 크기/좌표계 비교 ─────────────────────
    lH, lW, lT, lCRS = get_label_tif_info(label_path)
    bH, bW, bT, bCRS = get_band_info(scene_dir)

    print("[1] 공간 정보 비교")
    print(f"  label TIF  : ({lH} × {lW})  transform={lT[:6]}")
    print(f"  raw band   : ({bH} × {bW})  transform={bT[:6]}")
    offset_row = round((lT[5] - bT[5]) / bT[4])   # (y_origin_diff) / pixel_size
    offset_col = round((lT[2] - bT[2]) / bT[0])   # (x_origin_diff) / pixel_size
    print(f"  → 픽셀 오프셋  row={offset_row}  col={offset_col}")
    if offset_row != 0 or offset_col != 0:
        print("  *** 공간 오프셋 존재 → label_raw[i,j] ≠ band[i,j] ***")
    else:
        print("  공간 오프셋 없음 (좌표계 일치)")

    # ── 2. find_patch_coords 로 (row, col) 찾기 ───────────────────────
    print(f"\n[2] find_patch_coords로 PATCH{patch_idx} 위치 탐색...")
    row, col = find_patch_coords(scene_dir, label_path, patch_idx)
    print(f"  → row={row}, col={col}")

    # ── 3. zarr 내 저장된 label 불러오기 ──────────────────────────────
    store     = zarr.open_group(patch_path, mode='r')
    zarr_lbl  = store['label'][:].astype(np.uint8)
    print(f"\n[3] zarr label 분포: "
          + ", ".join(f"{v}:{(zarr_lbl==v).sum()}" for v in [0, 1, 2, 255]))

    # ── 4. label TIF에서 여러 위치 후보 비교 ──────────────────────────
    candidates = {
        "(row, col)":          (row, col),
        "(row, col-stride)":   (row, col - STRIDE),
        "(row, col+stride)":   (row, col + STRIDE),
        "(row-stride, col)":   (row - STRIDE, col),
        "(row+stride, col)":   (row + STRIDE, col),
    }

    print("\n[4] label TIF 위치 후보별 zarr label과 일치율:")
    best_score = -1
    best_key   = None
    for key, (r, c) in candidates.items():
        with rasterio.open(label_path) as src:
            H, W = src.height, src.width
        if r < 0 or c < 0 or r + PATCH_SIZE > H or c + PATCH_SIZE > W:
            print(f"  {key:25s}: 범위 초과 스킵")
            continue
        lbl_cand = load_label_patch(label_path, r, c)
        score = match_score(zarr_lbl, lbl_cand)
        marker = " ← BEST" if score > best_score else ""
        print(f"  {key:25s}: {score:.4f}{marker}")
        if score > best_score:
            best_score = score
            best_key   = key

    # ── 5. QA_PIXEL과도 비교 (Fmask 기준점 확인) ──────────────────────
    qa_file = find_qa_pixel_file(str(scene_dir))
    with rasterio.open(qa_file) as src:
        win = rwin.Window(col, row, PATCH_SIZE, PATCH_SIZE)
        qa  = src.read(1, window=win).astype(np.uint16)
    # QA cloud bits: 3(cloud), 4(shadow), 1(dilated), 2(cirrus)
    from utils.qa_pixel_mapping import qa_pixel_to_binary
    fmask = qa_pixel_to_binary(qa)
    score_fmask = match_score(zarr_lbl, fmask)
    print(f"\n[5] zarr label vs Fmask(QA_PIXEL at row,col): {score_fmask:.4f}")

    # ── 결론 ──────────────────────────────────────────────────────────
    print(f"\n=== 결론 ===")
    print(f"  zarr label과 가장 일치하는 label TIF 위치: {best_key}  (score={best_score:.4f})")
    if best_key == "(row, col)":
        print("  → 패치 저장 시 label 위치 정확. 시각화 코드 문제일 가능성 높음.")
    else:
        print("  → 패치 저장 시 label이 잘못된 위치에서 읽혔음.")
        print("    학습 데이터 자체에 GT 오프셋 버그 존재.")


if __name__ == '__main__':
    main()
