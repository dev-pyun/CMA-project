"""
라벨된 Landsat 씬 → 256×256 Zarr patch 분할.

학습 파이프라인과 완전히 동일한 포맷으로 저장하므로
patch_dataset.py 가 training/validation/test 패치를 구분 없이 로딩 가능.

입력  : scene_dir/   — 원본 Landsat L1 TIF 파일들 (B1~B7, B9, QA_PIXEL)
        label_path   — label_scene.py 출력 (H, W) uint8
            0=미라벨, 1=water, 2=snow, 3=shadow, 4=cloud, 255=fill

출력  : <out_root>/<scene_id>_PATCH{n}.zarr
            spectral/  (H,W,8)  uint16 — B1-B7, B9 raw DN
            rgb/       (H,W,3)  float32
            hsv/       (H,W,3)  float32
            sobel/     (H,W,3)  float32
            label/     (H,W)    uint8   — remap 후: 0=no-cloud, 1=cloud, 255=ignore

label remap:
    {1(water), 2(snow)}      → 0  (no-cloud)
    {3(shadow), 4(cloud)}    → 1  (cloud)
    {0(미라벨), 255(fill)}   → 255 (ignore)

필터링:
  - fill 비율 > 50%       → 버림
  - 유효 라벨 비율 < 5%  → 버림 (수동 라벨 패치는 val/test 전용이므로 관대하지 않게)

사용 예:
  # validation 패치 생성
  python scene_to_patches.py \\
      --scene_dir  /earth00_home/immj/Landsat/.../LC08_L1GT_188114_20201114_..._02_T2 \\
      --label_path labels/LC08_L1GT_188114_20201114_..._labels.tif \\
      --split val

  # test 패치 생성
  python scene_to_patches.py \\
      --scene_dir  /earth00_home/immj/Landsat/.../LC08_L1GT_... \\
      --label_path labels/LC08_L1GT_..._labels.tif \\
      --split test
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

# utils/ 가 있는 src/ 를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.split_scene import (
    ALL_BANDS, N_SPECTRAL,
    find_band_file, find_qa_pixel_file,
    compute_rgb, compute_hsv, compute_sobel, save_patch_zarr,
)
from utils.dir_paths import VALID_ZARR_PATH, TEST_ZARR_PATH


# label_scene.py 출력 → 학습 binary 형식
# {shadow(3), cloud(4)} → 1(cloud)
# {water(1), snow(2)}   → 0(no-cloud)
# {nodata(0), fill(255)} → 255(ignore)
LABEL_REMAP = {0: 255, 1: 0, 2: 0, 3: 1, 4: 1, 255: 255}
VALID_LABEL_VALUES = {1, 2, 3, 4}

DEFAULT_OUT = {
    'val':  VALID_ZARR_PATH,
    'test': TEST_ZARR_PATH,
}


def remap_labels(labels: np.ndarray) -> np.ndarray:
    out = np.full_like(labels, 255, dtype=np.uint8)
    for src_val, dst_val in LABEL_REMAP.items():
        out[labels == src_val] = dst_val
    return out


def process_scene(
    scene_dir: Path,
    label_path: Path,
    out_root: Path,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_frac: float = 0.05,
    max_fill_frac: float = 0.5,
) -> int:
    scene_id = scene_dir.name
    print(f"[처리 시작] {scene_id}")

    # 밴드 파일 확인
    band_files = {}
    for bk in ALL_BANDS:
        bf = find_band_file(str(scene_dir), bk)
        if bf:
            band_files[bk] = bf
        elif bk != 'B9':  # B1-B7 은 필수
            print(f"  [오류] 필수 밴드 {bk} 없음. 스킵.")
            return 0

    # QA_PIXEL (fill 마스크용)
    qa_file = find_qa_pixel_file(str(scene_dir))
    if qa_file is None:
        print("  [오류] QA_PIXEL 없음. 스킵.")
        return 0

    # 수동 라벨 로드
    with rasterio.open(label_path) as src:
        labels_raw = src.read(1).astype(np.uint8)

    print("  labels 원본 통계:")
    names = {0: "미라벨", 1: "water", 2: "snow", 3: "shadow", 4: "cloud", 255: "fill"}
    for v, name in names.items():
        pct = (labels_raw == v).mean() * 100
        if pct > 0:
            print(f"    {v:3d} {name:8s}: {pct:.1f}%")

    # 참조 크기
    with rasterio.open(list(band_files.values())[0]) as src:
        H, W = src.height, src.width
    print(f"  scene shape: ({H}, {W})")

    out_root.mkdir(parents=True, exist_ok=True)
    n_saved = n_skipped_fill = n_skipped_label = 0

    pbar = tqdm(total=((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1),
                desc=f"  {scene_id[:35]}", unit="patch", leave=False, ncols=100)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):

            lr = labels_raw[i:i + patch_size, j:j + patch_size]

            # fill 비율 체크 (QA_PIXEL Bit 0)
            with rasterio.open(qa_file) as src:
                from rasterio import windows as rwin
                win = rwin.Window(j, i, patch_size, patch_size)
                qa_patch = src.read(1, window=win).astype(np.uint16)
            fill_frac = float((qa_patch & 1).astype(bool).mean())
            if fill_frac > max_fill_frac:
                n_skipped_fill += 1
                pbar.update(1)
                continue

            # 유효 라벨 비율 체크
            valid_frac = float(np.isin(lr, list(VALID_LABEL_VALUES)).mean())
            if valid_frac < min_valid_frac:
                n_skipped_label += 1
                pbar.update(1)
                continue

            # 스펙트럴 밴드 읽기 → (H, W, 8) uint16
            spectral = np.zeros((patch_size, patch_size, N_SPECTRAL), dtype=np.uint16)
            for ch, bk in enumerate(ALL_BANDS):
                if bk not in band_files:
                    continue
                with rasterio.open(band_files[bk]) as src:
                    data = src.read(1, window=win)
                h, w = data.shape
                spectral[:h, :w, ch] = data

            # 파생 피처
            rgb   = compute_rgb(spectral)
            hsv   = compute_hsv(rgb)
            sobel = compute_sobel(rgb)

            # 라벨 remap
            label = remap_labels(lr)

            # Zarr 저장
            patch_path = str(out_root / f"{scene_id}_PATCH{n_saved}.zarr")
            save_patch_zarr(patch_path, spectral, rgb, hsv, sobel, label)

            n_saved += 1
            pbar.update(1)
            pbar.set_postfix(saved=n_saved)

    pbar.close()
    print(f"  ✓ {scene_id}: {n_saved} 저장, "
          f"{n_skipped_fill} 스킵(fill), {n_skipped_label} 스킵(라벨 부족)")
    print(f"    remap: {{1,2}}→0(no-cloud)  {{3,4}}→1(cloud)  {{0,255}}→255(ignore)")
    print(f"    출력: {out_root}")
    return n_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="라벨된 씬 → Zarr patch 분할 (학습 파이프라인 호환)")
    parser.add_argument("--scene_dir",  type=Path, required=True,
                        help="원본 Landsat L1 씬 폴더 (*_B1.TIF 등이 있는 곳)")
    parser.add_argument("--label_path", type=Path, required=True,
                        help="label_scene.py 출력 라벨 GeoTIFF")
    parser.add_argument("--split",      choices=["val", "test"], default="val",
                        help="val → VALIDATION_ZARR, test → TEST_ZARR")
    parser.add_argument("--out_root",   type=Path, default=None,
                        help="출력 디렉토리 직접 지정 (지정 시 --split 무시)")
    parser.add_argument("--patch_size", type=int,   default=256)
    parser.add_argument("--stride",     type=int,   default=256)
    parser.add_argument("--min_valid_frac", type=float, default=0.05)
    parser.add_argument("--max_fill_frac",  type=float, default=0.5)
    args = parser.parse_args()

    out_root = args.out_root if args.out_root else Path(DEFAULT_OUT[args.split])
    print(f"[출력 경로] {out_root}  (split={args.split})")

    process_scene(
        scene_dir=args.scene_dir,
        label_path=args.label_path,
        out_root=out_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_frac=args.min_valid_frac,
        max_fill_frac=args.max_fill_frac,
    )
