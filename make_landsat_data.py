"""
GeoTIFF → HDF5 conversion entry point for Landsat 8 scenes.

This script is a convenience wrapper that calls the scene splitting utility.

Usage:
    # Process training data
    python make_landsat_data.py --mode train

    # Process validation data
    python make_landsat_data.py --mode test

    # Process custom path with cirrus band
    python make_landsat_data.py --mode train --path /path/to/scenes --include_cirrus

Data to copy into src/data/TRAIN/ (per scene, one folder per date):
    Required: *_B1.TIF, *_B2.TIF, *_B3.TIF, *_B4.TIF,
              *_B5.TIF, *_B6.TIF, *_B7.TIF, *_QA_PIXEL.TIF
    Optional: *_B9.TIF (Cirrus, for cirrus_ndsi mode)
    Not needed: *_B8.TIF (Pan), *_B10.TIF, *_B11.TIF (Thermal),
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
        include_cirrus=args.include_cirrus,
    )
