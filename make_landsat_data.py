"""
GeoTIFF → HDF5 conversion entry point for Landsat 8 scenes.

This script is a convenience wrapper that calls the scene splitting utility.
All 8 spectral bands (B1–B7, B9) are always written to each H5 patch.
B9 (Cirrus) is zero-filled when absent in the source.

Usage:
    # Process training data (default path: src/data/TRAIN/)
    python make_landsat_data.py --mode train

    # Process from a custom path (supports nested year/month/date/scene structure)
    python make_landsat_data.py --mode train --path /path/to/scenes

    # Process validation data
    python make_landsat_data.py --mode test

Required files per scene directory:
    *_B1.TIF, *_B2.TIF, *_B3.TIF, *_B4.TIF,
    *_B5.TIF, *_B6.TIF, *_B7.TIF, *_QA_PIXEL.TIF
Optional:
    *_B9.TIF (Cirrus — zero-filled if absent, never skips a scene)
Not needed:
    *_B8.TIF (Pan), *_B10.TIF, *_B11.TIF (Thermal),
    *_MTL.*, *_ANG.txt, SAA/SZA/VAA/VZA, thumbnail files
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.split_scene import get_args, make_patches, setup_logger
from utils.dir_paths import (TRAIN_SAFE_PATH, VALID_SAFE_PATH, PRED_PATH,
                              TRAIN_PATH, VALID_PATH)


if __name__ == '__main__':
    setup_logger()
    args = get_args()

    # Set default paths based on mode
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

    make_patches(
        scene_parent_dir=os.path.abspath(args.path),
        out_folder=out_path,
        mode=args.mode,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )
