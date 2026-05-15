"""
TRAIN_ZARR 패치의 label 배열을 binary(0/1) → 3-class(0/1/2)로 in-place 업데이트.

spectral / rgb / hsv / sobel 은 그대로 유지하고,
QA_PIXEL TIF 를 재읽어 새로운 qa_pixel_to_binary() 결과로 label만 덮어쓴다.

split_scene.py 와 동일한 루프 순서(iy → ix)로 패치 번호를 재현하므로
기존 PATCH{n}.zarr 파일과 1:1 대응이 보장된다.

사용:
    conda run -n remote python update_train_labels.py
    conda run -n remote python update_train_labels.py --dry_run   # 패치 수만 확인
"""

import argparse
import glob
import os
import sys

import numpy as np
import zarr
from rasterio import open as raster_open
from rasterio import windows
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dir_paths import TRAIN_ZARR_PATH
from utils.split_scene import find_band_file, find_qa_pixel_file
from utils.qa_pixel_mapping import qa_pixel_to_binary, BINARY_NODATA

PATCH_SIZE = 256
OVERLAP    = 0
STEP       = PATCH_SIZE - OVERLAP
TRAIN_SCENE_DIR = '/home/pyuncb/src/data/TRAIN'   # 심볼릭 링크 모음


def get_scene_dirs():
    entries = sorted(os.listdir(TRAIN_SCENE_DIR))
    return [os.path.join(TRAIN_SCENE_DIR, e) for e in entries
            if os.path.isdir(os.path.join(TRAIN_SCENE_DIR, e))]


def update_scene_labels(scene_dir: str, zarr_out: str, dry_run: bool) -> dict:
    scene_name = os.path.basename(scene_dir)

    qa_file = find_qa_pixel_file(scene_dir)
    if qa_file is None:
        print(f'  [SKIP] {scene_name}: QA_PIXEL 없음')
        return {}

    ref_band = find_band_file(scene_dir, 'B5')
    if ref_band is None:
        print(f'  [SKIP] {scene_name}: B5 밴드 없음')
        return {}

    with raster_open(ref_band) as src:
        img_h, img_w = src.height, src.width

    n_patches_y = max(1, (img_h - OVERLAP) // STEP)
    n_patches_x = max(1, (img_w - OVERLAP) // STEP)

    n_saved   = 0   # 패치 번호 (skip된 것은 카운트 안 함)
    n_updated = 0
    n_missing = 0
    class_counts = np.zeros(3, dtype=np.int64)

    pbar = tqdm(total=n_patches_y * n_patches_x,
                desc=f'  {scene_name[:40]}', unit='patch',
                leave=False, ncols=100)

    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            row_start = max(0, min(iy * STEP, img_h - PATCH_SIZE))
            col_start = max(0, min(ix * STEP, img_w  - PATCH_SIZE))

            win = windows.Window(col_start, row_start, PATCH_SIZE, PATCH_SIZE)
            with raster_open(qa_file) as src:
                qa_data = src.read(1, window=win).astype(np.uint16)

            h, w = qa_data.shape
            qa_full = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
            qa_full[:h, :w] = qa_data

            new_label = qa_pixel_to_binary(qa_full)

            # split_scene.py 와 동일한 skip 조건
            if np.all(new_label == BINARY_NODATA):
                pbar.update(1)
                continue

            patch_path = os.path.join(zarr_out, f'{scene_name}_PATCH{n_saved}.zarr')

            if not os.path.exists(patch_path):
                n_missing += 1
                n_saved += 1
                pbar.update(1)
                continue

            if not dry_run:
                store = zarr.open_group(patch_path, mode='r+')
                store['label'][:] = new_label

            # 클래스별 픽셀 수 집계 (nodata 제외)
            for c in range(3):
                class_counts[c] += int((new_label == c).sum())

            n_updated += 1
            n_saved   += 1
            pbar.update(1)

    pbar.close()

    total_px = class_counts.sum()
    freq = class_counts / total_px if total_px > 0 else class_counts
    print(f'  {"[DRY]" if dry_run else "[DONE]"} {scene_name}: '
          f'{n_updated} 패치 {"확인" if dry_run else "업데이트"}'
          + (f', {n_missing} 패치 누락' if n_missing else '')
          + f'  |  no-cloud={freq[0]:.1%}  cloud={freq[1]:.1%}  shadow={freq[2]:.1%}')

    return {'updated': n_updated, 'missing': n_missing, 'class_counts': class_counts}


def main():
    parser = argparse.ArgumentParser(description='TRAIN_ZARR label binary → 3-class 업데이트')
    parser.add_argument('--dry_run', action='store_true',
                        help='실제 쓰기 없이 패치 수와 클래스 분포만 확인')
    args = parser.parse_args()

    mode_str = 'DRY RUN (쓰기 없음)' if args.dry_run else '실제 업데이트'
    print(f'\n{"="*60}')
    print(f'  TRAIN_ZARR label 3-class 업데이트  [{mode_str}]')
    print(f'  출력 경로: {TRAIN_ZARR_PATH}')
    print(f'{"="*60}\n')

    scene_dirs = get_scene_dirs()
    if not scene_dirs:
        print(f'씬 디렉토리를 찾을 수 없습니다: {TRAIN_SCENE_DIR}')
        sys.exit(1)

    print(f'씬 수: {len(scene_dirs)}\n')

    total_updated = 0
    total_missing = 0
    total_counts  = np.zeros(3, dtype=np.int64)

    for scene_dir in scene_dirs:
        result = update_scene_labels(scene_dir, TRAIN_ZARR_PATH, args.dry_run)
        if result:
            total_updated += result['updated']
            total_missing += result['missing']
            total_counts  += result['class_counts']

    total_px = total_counts.sum()
    freq = total_counts / total_px if total_px > 0 else total_counts

    print(f'\n{"="*60}')
    print(f'{"완료" if not args.dry_run else "DRY RUN 완료"}')
    print(f'  총 {"업데이트" if not args.dry_run else "확인"} 패치: {total_updated:,}')
    if total_missing:
        print(f'  누락 패치: {total_missing}')
    print(f'  전체 클래스 분포 (nodata 제외):')
    print(f'    no-cloud : {freq[0]:.2%}  ({total_counts[0]:,} px)')
    print(f'    cloud    : {freq[1]:.2%}  ({total_counts[1]:,} px)')
    print(f'    shadow   : {freq[2]:.2%}  ({total_counts[2]:,} px)')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
