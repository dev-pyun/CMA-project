"""
Join patch-level predictions back into full scene GeoTIFF files.

Usage:
    python -m utils.join_predictions \
        --h5_dir /path/to/prediction_h5/ \
        --meta_file /path/to/scene_META.npy \
        --output /path/to/output.tif
"""

import argparse
import glob
import logging
import os

import h5py
import numpy as np
from rasterio import open as raster_open
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def join_patches(h5_dir, output_path=None, prediction_channel=-1):
    """
    Join HDF5 patch predictions into a single numpy array.

    Parameters
    ----------
    h5_dir : str
        Directory containing PATCH*.h5 files.
    output_path : str, optional
        If provided, save as .npy file.
    prediction_channel : int
        Channel index in the H5 data that contains predictions.
        Default -1 (last channel).

    Returns
    -------
    patches : list of dict
        Each dict has 'data' (np.ndarray) and 'index' (int).
    """
    h5_files = sorted(glob.glob(os.path.join(h5_dir, '*PATCH*.h5')))
    if not h5_files:
        raise FileNotFoundError(f'No H5 patch files found in {h5_dir}')

    patches = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as hf:
            data = hf.get('data')[:]
            pred = data[:, :, prediction_channel]

        # Extract patch index from filename
        basename = os.path.basename(h5_path)
        idx = int(basename.split('PATCH')[1].split('.')[0])
        patches.append({'data': pred, 'index': idx})

    patches.sort(key=lambda x: x['index'])

    if output_path:
        all_preds = np.stack([p['data'] for p in patches])
        np.save(output_path, all_preds)
        logger.info(f'Saved joined predictions to {output_path}')

    return patches


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Join patch predictions into a scene')
    parser.add_argument('--h5_dir', required=True,
                        help='Directory with PATCH*.h5 files')
    parser.add_argument('--output', default=None,
                        help='Output .npy file path')
    parser.add_argument('--channel', type=int, default=-1,
                        help='Prediction channel index')
    args = parser.parse_args()

    join_patches(args.h5_dir, args.output, args.channel)
