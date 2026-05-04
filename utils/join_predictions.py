"""
Join patch-level predictions back into full scene GeoTIFF files.

Usage:
    python -m utils.join_predictions \
        --zarr_dir /path/to/prediction_zarr/ \
        --output /path/to/output.npy \
        --key raw_prediction
"""

import argparse
import glob
import logging
import os

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def join_patches(zarr_dir, output_path=None, key='raw_prediction'):
    """
    Join Zarr patch predictions into a list ordered by patch index.

    Parameters
    ----------
    zarr_dir : str
        Directory containing PATCH*.zarr directories.
    output_path : str, optional
        If provided, save stacked predictions as .npy file.
    key : str
        Array name inside each zarr patch to read (e.g. 'raw_prediction',
        'pseudo_label', or 'label').

    Returns
    -------
    patches : list of dict
        Each dict has 'data' (np.ndarray) and 'index' (int), sorted by index.
    """
    zarr_files = sorted(glob.glob(os.path.join(zarr_dir, '*PATCH*.zarr')))
    if not zarr_files:
        raise FileNotFoundError(f'No .zarr patch files found in {zarr_dir}')

    patches = []
    for zarr_path in zarr_files:
        store = zarr.open_group(zarr_path, mode='r')
        if key not in store:
            logger.warning(f'Key "{key}" not found in {zarr_path}, skipping.')
            continue
        data = store[key][:]

        basename = os.path.basename(zarr_path)
        idx = int(basename.split('PATCH')[1].split('.')[0])
        patches.append({'data': data, 'index': idx})

    patches.sort(key=lambda x: x['index'])

    if output_path:
        all_preds = np.stack([p['data'] for p in patches])
        np.save(output_path, all_preds)
        logger.info(f'Saved joined predictions to {output_path}')

    return patches


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Join zarr patch predictions into a scene')
    parser.add_argument('--zarr_dir', required=True,
                        help='Directory with PATCH*.zarr directories')
    parser.add_argument('--output', default=None,
                        help='Output .npy file path')
    parser.add_argument('--key', default='raw_prediction',
                        help='Zarr array key to read (default: raw_prediction)')
    args = parser.parse_args()

    join_patches(args.zarr_dir, args.output, args.key)
