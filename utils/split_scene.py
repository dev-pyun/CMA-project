"""
Split Landsat 8 scenes (per-band TIF files) into fixed-size HDF5 patches
for network training.

Landsat 8 data layout (per scene directory):
    *_B1.TIF  .. *_B7.TIF, *_B9.TIF   (spectral bands, 30m)
    *_QA_PIXEL.TIF                     (quality assessment)
    *_MTL.json                         (metadata)

Output HDF5 layout  (256 × 256 patches):
    data[:, :, 0]  → B1  (Coastal)
    data[:, :, 1]  → B2  (Blue)
    data[:, :, 2]  → B3  (Green)
    data[:, :, 3]  → B4  (Red)
    data[:, :, 4]  → B5  (NIR)
    data[:, :, 5]  → B6  (SWIR1)
    data[:, :, 6]  → B7  (SWIR2)
    data[:, :, 7]  → B9  (Cirrus, zero-filled if absent in source)
    data[:, :, 8]  → QA_PIXEL labels  (remapped to 0–5)
"""

import argparse
import glob
import logging
import os
import random
import warnings
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from cv2 import resize, INTER_NEAREST, INTER_CUBIC
from tqdm import tqdm

# Suppress PROJ database version warnings from rasterio
warnings.filterwarnings('ignore', message='.*PROJ.*')
os.environ['PROJ_LIB'] = ''  # Prevent PROJ from searching extra paths
logging.getLogger('rasterio').setLevel(logging.ERROR)

from rasterio import open as raster_open
from rasterio import windows

from utils.qa_pixel_mapping import qa_pixel_to_classes

logger = logging.getLogger(__name__)

# Required spectral bands — scene is skipped if any are missing
REQUIRED_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

# All bands written to H5 (order = channel index); B9 is zero-filled if absent
ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']
N_SPECTRAL = len(ALL_BANDS)  # always 8


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Split Landsat 8 scenes into HDF5 patches')
    parser.add_argument('-p', '--path',
                        help='Path to folder containing scene directories',
                        default=None)
    parser.add_argument('-o', '--out_path',
                        help='Explicit output path for HDF5 patches (overrides default)',
                        default=None)
    parser.add_argument('-m', '--mode', default='train',
                        choices=['train', 'test', 'predict'],
                        help='Processing mode')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size in pixels (default: 256)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Overlap between patches in pixels')
    return parser.parse_args(argv)


def find_band_file(scene_dir, band_key):
    """Find the TIF file for a specific band in a scene directory."""
    pattern = os.path.join(scene_dir, f'*_{band_key}.TIF')
    matches = glob.glob(pattern)
    if not matches:
        # Try lowercase
        pattern = os.path.join(scene_dir, f'*_{band_key}.tif')
        matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_qa_pixel_file(scene_dir):
    """Find the QA_PIXEL TIF file in a scene directory."""
    pattern = os.path.join(scene_dir, '*_QA_PIXEL.TIF')
    matches = glob.glob(pattern)
    if not matches:
        pattern = os.path.join(scene_dir, '*_QA_PIXEL.tif')
        matches = glob.glob(pattern)
    return matches[0] if matches else None


def read_band(filepath, window=None):
    """Read a single band from a GeoTIFF file."""
    with raster_open(filepath) as src:
        data = src.read(1, window=window)
        transform = src.transform if window is None else \
            windows.transform(window, src.transform)
        bounds = src.bounds
    return data, transform, bounds


def split_scene_to_patches(scene_dir, out_folder, mode='train',
                           patch_size=256, overlap=0):
    """
    Split a single Landsat 8 scene into overlapping patches saved as HDF5.

    Parameters
    ----------
    scene_dir : str
        Path to the scene directory containing per-band TIF files.
    out_folder : str
        Output directory for HDF5 patch files.
    mode : str
        'train', 'test', or 'predict'.
    patch_size : int
        Size of each square patch (pixels).
    overlap : int
        Overlap between adjacent patches (pixels).
    """
    scene_name = os.path.basename(scene_dir)

    # Find required band files (B1–B7); scene is skipped if any are missing
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

    # B9 (Cirrus): always attempted; zero-filled if absent
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

    # Read reference band (B5) to get dimensions and georef
    ref_band_key = 'B5'
    ref_path = band_files[ref_band_key]
    with raster_open(ref_path) as ref:
        img_height = ref.height
        img_width = ref.width

    # Always N_SPECTRAL (8) band channels + 1 QA_PIXEL label channel
    n_channels = N_SPECTRAL + (1 if qa_file else 0)

    # Compute patch grid
    step = patch_size - overlap
    n_patches_y = max(1, (img_height - overlap) // step)
    n_patches_x = max(1, (img_width - overlap) // step)
    total_patches = n_patches_y * n_patches_x

    os.makedirs(out_folder, exist_ok=True)
    n_saved = 0
    n_skipped = 0

    # Progress bar for patches within this scene
    patch_pbar = tqdm(
        total=total_patches,
        desc=f'  {scene_name[:40]}',
        unit='patch',
        leave=False,
        ncols=100,
    )

    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            row_start = min(iy * step, img_height - patch_size)
            col_start = min(ix * step, img_width - patch_size)
            row_start = max(0, row_start)
            col_start = max(0, col_start)

            win = windows.Window(col_start, row_start, patch_size, patch_size)

            # Read all bands into a (H, W, C) array
            patch_data = np.zeros((patch_size, patch_size, n_channels),
                                 dtype=np.uint16)

            for ch, bk in enumerate(ALL_BANDS):
                if bk not in band_files:
                    continue  # zero-fill (already zeros from np.zeros)
                with raster_open(band_files[bk]) as src:
                    data = src.read(1, window=win)
                h, w = data.shape
                patch_data[:h, :w, ch] = data

            # QA_PIXEL → 6-class labels (always at channel N_SPECTRAL)
            if qa_file:
                with raster_open(qa_file) as src:
                    qa_data = src.read(1, window=win)
                h, w = qa_data.shape
                labels = qa_pixel_to_classes(qa_data)
                patch_data[:h, :w, N_SPECTRAL] = labels

                # Skip patches that contain ANY fill (no-data) pixel
                if np.any(labels == 0):
                    n_skipped += 1
                    patch_pbar.update(1)
                    continue

            # Save as HDF5
            patch_name = f'{scene_name}_PATCH{n_saved}.h5'
            patch_path = os.path.join(out_folder, patch_name)
            with h5py.File(patch_path, 'w') as hf:
                hf.create_dataset('data', data=patch_data)

            n_saved += 1
            patch_pbar.update(1)
            patch_pbar.set_postfix(saved=n_saved, skipped=n_skipped)

    patch_pbar.close()
    print(f'  ✓ {scene_name}: {n_saved} saved, {n_skipped} skipped (no-data), '
          f'{img_height}x{img_width} px')
    return n_saved


def make_patches(scene_parent_dir, out_folder, mode='train',
                 patch_size=256, overlap=0):
    """
    Process all scene directories under a parent directory.

    Each scene directory should contain per-band TIF files
    (e.g., LC08_..._B1.TIF, LC08_..._B2.TIF, ...).
    """
    # Find scene directories: look for directories containing B1.TIF
    scene_dirs = []
    # First check if current directory IS a scene (flat structure)
    if find_band_file(scene_parent_dir, 'B1') is not None:
        scene_dirs = [scene_parent_dir]
    else:
        # Look for subdirectories containing band files
        for root, dirs, files in os.walk(scene_parent_dir):
            if find_band_file(root, 'B1') is not None:
                scene_dirs.append(root)
                dirs.clear()

    if not scene_dirs:
        raise FileNotFoundError(
            f'No Landsat 8 scenes found in {scene_parent_dir}. '
            f'Expected directories containing *_B1.TIF files.')

    total = len(scene_dirs)
    print(f'\n=== Landsat 8 Scene Processor ===')
    print(f'Found {total} scene(s) to process')
    print(f'Patch size: {patch_size}x{patch_size}, Overlap: {overlap}')
    print(f'Output: {out_folder}')
    print(f'Cirrus (B9): always included (zero-filled if absent)')
    print(f'================================\n')

    total_patches = 0
    scene_pbar = tqdm(
        scene_dirs,
        desc='Scenes',
        unit='scene',
        ncols=100,
    )

    for scene_dir in scene_pbar:
        scene_name = os.path.basename(scene_dir)
        scene_pbar.set_description(f'Scene: {scene_name[:30]}')
        n = split_scene_to_patches(
            scene_dir, out_folder, mode=mode,
            patch_size=patch_size, overlap=overlap)
        if n:
            total_patches += n
        scene_pbar.set_postfix(total_patches=total_patches)

    print(f'\n=== Done! ===')
    print(f'Total scenes processed: {total}')
    print(f'Total patches created:  {total_patches}')
    print(f'Output directory:       {out_folder}')


if __name__ == '__main__':
    random.seed(42)
    setup_logger()

    args = get_args()

    from utils.dir_paths import TRAIN_SAFE_PATH, VALID_SAFE_PATH, PRED_PATH
    from utils.dir_paths import TRAIN_PATH, VALID_PATH

    if args.path is None:
        if args.mode == 'train':
            args.path = TRAIN_SAFE_PATH
            out_path = TRAIN_PATH
        elif args.mode == 'test':
            args.path = VALID_SAFE_PATH
            out_path = VALID_PATH
        elif args.mode == 'predict':
            args.path = PRED_PATH
            out_path = args.path + '_H5'
    else:
        out_path = args.path + '_H5'

    if getattr(args, 'out_path', None) is not None:
        out_path = args.out_path

    make_patches(
        scene_parent_dir=os.path.abspath(args.path),
        out_folder=out_path,
        mode=args.mode,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )
