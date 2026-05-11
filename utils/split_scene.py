"""
Split Landsat 8 scenes (per-band TIF files) into fixed-size Zarr patches
for network training.

Landsat 8 data layout (per scene directory):
    *_B1.TIF  .. *_B7.TIF, *_B9.TIF   (spectral bands, 30m)
    *_QA_PIXEL.TIF                     (quality assessment)
    *_MTL.json                         (metadata)

Output Zarr layout  (256 × 256 patches, one .zarr directory per patch):
    spectral  (H, W, 8)  uint16   — B1–B7, B9 raw DN values
    rgb       (H, W, 3)  float32  — percentile-normalised R(B4), G(B3), B(B2) ∈ [0,1]
    hsv       (H, W, 3)  float32  — H, S, V derived from rgb ∈ [0,1]
    sobel     (H, W, 3)  float32  — Sobel X, Y, Magnitude on luminance
    label     (H, W)     uint8    — binary: 0=no-cloud, 1=cloud, 255=no-data
"""

import argparse
import glob
import logging
import os
import random
import shutil
import warnings
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm

# Suppress PROJ database version warnings from rasterio
warnings.filterwarnings('ignore', message='.*PROJ.*')
os.environ['PROJ_LIB'] = ''
logging.getLogger('rasterio').setLevel(logging.ERROR)

from rasterio import open as raster_open
from rasterio import windows

from utils.qa_pixel_mapping import qa_pixel_to_binary, BINARY_NODATA

logger = logging.getLogger(__name__)

# Required spectral bands — scene is skipped if any are missing
REQUIRED_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

# All bands written to zarr (order = channel index); B9 is zero-filled if absent
ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']
N_SPECTRAL = len(ALL_BANDS)  # always 8

# Zarr compressors (zarr v3 BloscCodec)
_COMP_UINT16 = BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')
_COMP_F32    = BloscCodec(cname='zstd', clevel=3, shuffle='shuffle')
_COMP_UINT8  = BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Split Landsat 8 scenes into Zarr patches')
    parser.add_argument('-p', '--path',
                        help='Path to folder containing scene directories',
                        default=None)
    parser.add_argument('-o', '--out_path',
                        help='Explicit output path for Zarr patches (overrides default)',
                        default=None)
    parser.add_argument('-m', '--mode', default='train',
                        choices=['train', 'test', 'predict'],
                        help='Processing mode')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size in pixels (default: 256)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Overlap between patches in pixels')
    return parser.parse_args(argv)


# ── Band / file helpers ────────────────────────────────────────────────

def find_band_file(scene_dir, band_key):
    """Find the TIF file for a specific band in a scene directory."""
    for ext in ('TIF', 'tif'):
        matches = glob.glob(os.path.join(scene_dir, f'*_{band_key}.{ext}'))
        if matches:
            return matches[0]
    return None


def find_qa_pixel_file(scene_dir):
    """Find the QA_PIXEL TIF file in a scene directory."""
    for ext in ('TIF', 'tif'):
        matches = glob.glob(os.path.join(scene_dir, f'*_QA_PIXEL.{ext}'))
        if matches:
            return matches[0]
    return None


# ── Derived feature computation ────────────────────────────────────────

def _percentile_normalize(arr: np.ndarray, p_lo: int = 2, p_hi: int = 98) -> np.ndarray:
    """Normalize a 2-D band to [0, 1] using percentile clipping."""
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def compute_rgb(spectral: np.ndarray) -> np.ndarray:
    """Spectral (H,W,8) uint16 → percentile-normalised RGB float32 [0,1].

    Channel mapping: R=B4 (ch3), G=B3 (ch2), B=B2 (ch1).
    """
    r = _percentile_normalize(spectral[:, :, 3])  # B4
    g = _percentile_normalize(spectral[:, :, 2])  # B3
    b = _percentile_normalize(spectral[:, :, 1])  # B2
    return np.stack([r, g, b], axis=-1)


def compute_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB float32 [0,1] → HSV float32 [0,1].

    OpenCV output: H∈[0,180], S∈[0,255], V∈[0,255] — normalised here.
    """
    rgb_u8 = (rgb * 255).astype(np.uint8)
    hsv_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hsv = hsv_u8.astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv


def compute_sobel(rgb: np.ndarray) -> np.ndarray:
    """RGB float32 [0,1] → Sobel X, Y, Magnitude on luminance.

    Returns (H, W, 3) float32: [Sobel_X, Sobel_Y, Sobel_Magnitude].
    """
    gray = (0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(np.float32)
    sx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return np.stack([sx, sy, mag], axis=-1)


# ── Zarr I/O ───────────────────────────────────────────────────────────

def save_patch_zarr(patch_path: str, spectral: np.ndarray, rgb: np.ndarray,
                    hsv: np.ndarray, sobel: np.ndarray, label: np.ndarray):
    """Persist one patch to a zarr directory store."""
    store = zarr.open_group(patch_path, mode='w')
    store.create_array('spectral', data=spectral,
                       chunks=spectral.shape, compressors=[_COMP_UINT16])
    store.create_array('rgb',      data=rgb,
                       chunks=rgb.shape,      compressors=[_COMP_F32])
    store.create_array('hsv',      data=hsv,
                       chunks=hsv.shape,      compressors=[_COMP_F32])
    store.create_array('sobel',    data=sobel,
                       chunks=sobel.shape,    compressors=[_COMP_F32])
    store.create_array('label',    data=label,
                       chunks=label.shape,    compressors=[_COMP_UINT8])


# ── Main splitting logic ───────────────────────────────────────────────

def split_scene_to_patches(scene_dir, out_folder, mode='train',
                           patch_size=256, overlap=0):
    """
    Split a single Landsat 8 scene into overlapping patches saved as Zarr.

    Parameters
    ----------
    scene_dir : str
        Path to the scene directory containing per-band TIF files.
    out_folder : str
        Output directory for Zarr patch directories.
    mode : str
        'train', 'test', or 'predict'.
    patch_size : int
        Size of each square patch (pixels).
    overlap : int
        Overlap between adjacent patches (pixels).
    """
    scene_name = os.path.basename(scene_dir)
    done_marker = os.path.join(out_folder, f'{scene_name}.done')

    # 완료된 씬은 스킵 (-1 반환으로 호출자가 구분)
    if os.path.exists(done_marker):
        with open(done_marker) as f:
            n_prev = int(f.read().strip())
        logger.info(f'  {scene_name}: already done ({n_prev} patches), skipping.')
        return -1

    # 중간에 중단된 패치가 있으면 삭제 후 재처리
    partial = glob.glob(os.path.join(out_folder, f'{scene_name}_PATCH*.zarr'))
    if partial:
        logger.info(f'  {scene_name}: removing {len(partial)} partial patches, reprocessing.')
        for p in partial:
            shutil.rmtree(p)

    # Find required band files; scene is skipped if any are missing
    band_files = {}
    for bk in REQUIRED_BANDS:
        bf = find_band_file(scene_dir, bk)
        if bf is None:
            logger.warning(f'  Band {bk} not found in {scene_name}.')
        else:
            band_files[bk] = bf

    if len(band_files) < len(REQUIRED_BANDS):
        logger.error(f'  Missing core bands in {scene_name}, skipping scene.')
        return 0

    # B9 (Cirrus): zero-filled if absent
    b9_file = find_band_file(scene_dir, 'B9')
    if b9_file is not None:
        band_files['B9'] = b9_file
    else:
        logger.info(f'  B9 not found in {scene_name}, zero-filling channel 7.')

    # QA_PIXEL
    qa_file = find_qa_pixel_file(scene_dir)
    if qa_file is None and mode == 'train':
        logger.error(f'  QA_PIXEL not found in {scene_name}, skipping (train mode).')
        return 0

    # Reference dimensions from B5
    with raster_open(band_files['B5']) as ref:
        img_height = ref.height
        img_width  = ref.width

    step        = patch_size - overlap
    n_patches_y = max(1, (img_height - overlap) // step)
    n_patches_x = max(1, (img_width  - overlap) // step)
    total_patches = n_patches_y * n_patches_x

    os.makedirs(out_folder, exist_ok=True)
    n_saved   = 0
    n_skipped = 0

    patch_pbar = tqdm(
        total=total_patches,
        desc=f'  {scene_name[:40]}',
        unit='patch',
        leave=False,
        ncols=100,
    )

    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            row_start = max(0, min(iy * step, img_height - patch_size))
            col_start = max(0, min(ix * step, img_width  - patch_size))
            win = windows.Window(col_start, row_start, patch_size, patch_size)

            # Read spectral bands → (H, W, 8) uint16
            spectral = np.zeros((patch_size, patch_size, N_SPECTRAL), dtype=np.uint16)
            for ch, bk in enumerate(ALL_BANDS):
                if bk not in band_files:
                    continue
                with raster_open(band_files[bk]) as src:
                    data = src.read(1, window=win)
                h, w = data.shape
                spectral[:h, :w, ch] = data

            # QA_PIXEL → binary label
            if qa_file:
                with raster_open(qa_file) as src:
                    qa_data = src.read(1, window=win)
                h, w = qa_data.shape
                qa_full = np.zeros((patch_size, patch_size), dtype=np.uint16)
                qa_full[:h, :w] = qa_data
                label = qa_pixel_to_binary(qa_full)

                # fill pixels remain as 255 (ignored in loss via ignore_index=255)
            else:
                label = np.zeros((patch_size, patch_size), dtype=np.uint8)

            # Compute derived features
            rgb   = compute_rgb(spectral)
            hsv   = compute_hsv(rgb)
            sobel = compute_sobel(rgb)

            # Save as zarr
            patch_name = f'{scene_name}_PATCH{n_saved}.zarr'
            patch_path = os.path.join(out_folder, patch_name)
            save_patch_zarr(patch_path, spectral, rgb, hsv, sobel, label)

            n_saved += 1
            patch_pbar.update(1)
            patch_pbar.set_postfix(saved=n_saved, skipped=n_skipped)

    patch_pbar.close()
    print(f'  ✓ {scene_name}: {n_saved} saved, {n_skipped} skipped (no-data), '
          f'{img_height}x{img_width} px')

    # 완료 마커 기록
    with open(done_marker, 'w') as f:
        f.write(str(n_saved))

    return n_saved


def make_patches(scene_parent_dir, out_folder, mode='train',
                 patch_size=256, overlap=0):
    """
    Process all scene directories under a parent directory.

    Each scene directory should contain per-band TIF files
    (e.g., LC08_..._B1.TIF, LC08_..._B2.TIF, ...).
    """
    scene_dirs = []
    if find_band_file(scene_parent_dir, 'B1') is not None:
        scene_dirs = [scene_parent_dir]
    else:
        for root, dirs, files in os.walk(scene_parent_dir, followlinks=True):
            if find_band_file(root, 'B1') is not None:
                scene_dirs.append(root)
                dirs.clear()

    if not scene_dirs:
        raise FileNotFoundError(
            f'No Landsat 8 scenes found in {scene_parent_dir}. '
            f'Expected directories containing *_B1.TIF files.')

    total = len(scene_dirs)
    print(f'\n=== Landsat 8 Scene Processor (Zarr output) ===')
    print(f'Found {total} scene(s) to process')
    print(f'Patch size: {patch_size}x{patch_size}, Overlap: {overlap}')
    print(f'Output: {out_folder}')
    print(f'Label: binary (0=no-cloud, 1=cloud, 255=no-data)')
    print(f'Derived features: RGB, HSV, Sobel (X/Y/Mag)')
    print(f'================================================\n')

    total_patches = 0
    n_skipped_scenes = 0
    scene_pbar = tqdm(scene_dirs, desc='Scenes', unit='scene', ncols=100)

    for scene_dir in scene_pbar:
        scene_name = os.path.basename(scene_dir)
        scene_pbar.set_description(f'Scene: {scene_name[:30]}')
        n = split_scene_to_patches(
            scene_dir, out_folder, mode=mode,
            patch_size=patch_size, overlap=overlap)
        if n > 0:
            total_patches += n
        elif n < 0:
            n_skipped_scenes += 1
        scene_pbar.set_postfix(total_patches=total_patches, skipped=n_skipped_scenes)

    print(f'\n=== Done! ===')
    print(f'Total scenes:          {total}')
    print(f'  - processed:         {total - n_skipped_scenes}')
    print(f'  - skipped (done):    {n_skipped_scenes}')
    print(f'Total patches created: {total_patches}')
    print(f'Output directory:      {out_folder}')


if __name__ == '__main__':
    random.seed(42)
    setup_logger()

    args = get_args()

    from utils.dir_paths import TRAIN_ZARR_PATH, VALID_ZARR_PATH, PRED_PATH
    from utils.dir_paths import TRAIN_PATH, VALID_PATH
    from utils.dir_paths import TRAIN_SAFE_PATH, VALID_SAFE_PATH

    if args.path is None:
        if args.mode == 'train':
            args.path = TRAIN_SAFE_PATH
            out_path = TRAIN_ZARR_PATH
        elif args.mode == 'test':
            args.path = VALID_SAFE_PATH
            out_path = VALID_ZARR_PATH
        elif args.mode == 'predict':
            args.path = PRED_PATH
            out_path = args.path + '_ZARR'
    else:
        out_path = args.path + '_ZARR'

    if getattr(args, 'out_path', None) is not None:
        out_path = args.out_path

    make_patches(
        scene_parent_dir=os.path.abspath(args.path),
        out_folder=out_path,
        mode=args.mode,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )
