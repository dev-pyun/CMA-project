"""
utils/compute_global_stats.py — Global per-band spectral statistics + PCA
from all Landsat scenes under --root (default: Weddell Sea full archive).

Pass 1 (default): accumulate mean/std with Chan's algorithm → global_spectral_stats.npz
Pass 2 (--pca_only): load existing stats, fit IncrementalPCA on z-scored pixels → global_pca.npz

Fill pixels where any band value == 0 are excluded.

Outputs:
  data/global_spectral_stats.npz
    mean  : (8,) float64 — per-band mean reflectance  (÷10000 scale)
    std   : (8,) float64 — per-band std  reflectance  (÷10000 scale)
    count : int           — total valid pixels used
    bands : str array     — ['B1','B2','B3','B4','B5','B6','B7','B9']
    n_scenes : int        — number of scenes processed

  data/global_pca.npz
    components             : (8, 8) float64 — PCA eigenvectors (row = PC)
    explained_variance_ratio : (8,) float64
    mean                   : (8,) float64 — global band mean  (same as stats)
    std                    : (8,) float64 — global band std   (same as stats)

Usage:
    # Pass 1 — compute mean/std
    conda run -n remote python utils/compute_global_stats.py \\
        --root /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea \\
        --max_size 300

    # Pass 2 — fit global PCA (loads existing stats, ~58 min)
    conda run -n remote python utils/compute_global_stats.py \\
        --root /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea \\
        --max_size 300 --pca_only
"""

import argparse
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.split_scene import ALL_BANDS, N_SPECTRAL, find_band_file, \
    _dn_to_toa_uint16, _load_sun_sin
from utils.dir_paths import WEDDELL_SEA_SOURCE_PATH

BAND_LABELS  = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']
DEFAULT_OUT  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'global_spectral_stats.npz')
DEFAULT_PCA_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'global_pca.npz')


# ── Scene discovery ────────────────────────────────────────────────────

def find_scenes(root: str) -> list[str]:
    scenes = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.endswith('_B1.TIF') for f in filenames):
            scenes.append(dirpath)
    return sorted(scenes)


# ── Loading ────────────────────────────────────────────────────────────

def load_scene(scene_dir: str, max_size: int) -> np.ndarray | None:
    """
    Load TOA-converted spectral bands downsampled to max_size.
    Returns (H, W, 8) uint16 or None if a required band is missing.
    """
    band_files = {}
    for bk in ALL_BANDS:
        bf = find_band_file(scene_dir, bk)
        if bf:
            band_files[bk] = bf
        elif bk != 'B9':
            return None

    with rasterio.open(list(band_files.values())[0]) as src:
        H, W = src.height, src.width

    scale   = max(1, max(H, W) // max_size)
    h_out   = H // scale
    w_out   = W // scale
    spectral = np.zeros((h_out, w_out, N_SPECTRAL), dtype=np.uint16)

    for ch, bk in enumerate(ALL_BANDS):
        if bk not in band_files:
            continue
        with rasterio.open(band_files[bk]) as src:
            spectral[:, :, ch] = src.read(
                1, out_shape=(h_out, w_out),
                resampling=Resampling.average)

    try:
        sun_sin  = _load_sun_sin(scene_dir)
        spectral = _dn_to_toa_uint16(spectral, sun_sin=sun_sin)
    except Exception:
        return None

    return spectral


# ── Chan's parallel stats accumulation ────────────────────────────────

def update_stats(mean: np.ndarray, M2: np.ndarray, n: int,
                 X_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine running (mean, M2, n) with a new batch using Chan's algorithm."""
    m = len(X_batch)
    if m == 0:
        return mean, M2, n
    bm    = X_batch.mean(axis=0)
    bv    = X_batch.var(axis=0)
    delta = bm - mean
    new_n = n + m
    mean  = (n * mean + m * bm) / new_n
    M2   += bv * m + delta ** 2 * (n * m / new_n)
    return mean, M2, new_n


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute global per-band spectral statistics + PCA '
                    'from all Landsat scenes under --root')
    parser.add_argument('--root', default=WEDDELL_SEA_SOURCE_PATH,
                        help='Root folder to scan (default: Weddell Sea archive)')
    parser.add_argument('--max_size', type=int, default=300,
                        help='Max pixels on longest side per scene (default: 300)')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='Output .npz path for mean/std stats')
    parser.add_argument('--pca_out', default=DEFAULT_PCA_OUT,
                        help='Output .npz path for global PCA model')
    parser.add_argument('--pca_only', action='store_true',
                        help='Skip stats computation; load existing --out and fit PCA only')
    args = parser.parse_args()

    print(f'Root     : {args.root}')
    print(f'Max_size : {args.max_size}')
    print('Scanning scenes...')
    scenes = find_scenes(args.root)
    print(f'Found {len(scenes):,} scenes.\n')

    # ── Pass 1: mean / std ─────────────────────────────────────────────
    if args.pca_only:
        print(f'[--pca_only] Loading existing stats from {args.out}')
        d    = np.load(args.out)
        mean = d['mean'].astype(np.float64)
        std  = d['std'].astype(np.float64)
        print('Loaded. Skipping stats computation.\n')
    else:
        print(f'Out (stats) : {args.out}')
        mean    = np.zeros(8, dtype=np.float64)
        M2      = np.zeros(8, dtype=np.float64)
        n_total = 0
        n_skip  = 0

        t0 = time.time()
        for scene_dir in tqdm(scenes, desc='Stats pass'):
            sp = load_scene(scene_dir, args.max_size)
            if sp is None:
                n_skip += 1
                continue

            X      = sp.astype(np.float64) / 10000.0
            X_flat = X.reshape(-1, 8)
            valid  = (X_flat > 0).all(axis=1)
            X_v    = X_flat[valid]

            mean, M2, n_total = update_stats(mean, M2, n_total, X_v)

        elapsed = time.time() - t0
        std     = np.sqrt(M2 / n_total)

        print(f'\nProcessed scenes   : {len(scenes) - n_skip:,}')
        print(f'Skipped (bad data) : {n_skip}')
        print(f'Total valid pixels : {n_total:,}')
        print(f'Elapsed            : {elapsed / 60:.1f} min')
        print(f'\n{"Band":<6} {"Mean":>12} {"Std":>12}')
        print('-' * 32)
        for i, lbl in enumerate(BAND_LABELS):
            print(f'{lbl:<6} {mean[i]:>12.6f} {std[i]:>12.6f}')

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, mean=mean, std=std,
                 count=n_total, bands=BAND_LABELS,
                 n_scenes=len(scenes) - n_skip)
        print(f'\nSaved → {args.out}')

    # ── Pass 2: global PCA fit (IncrementalPCA on z-scored pixels) ──────
    print(f'\nOut (PCA)   : {args.pca_out}')
    print('Fitting global PCA (IncrementalPCA, z-scored pixels)...\n')
    std_safe = np.where(std < 1e-9, 1.0, std)
    ipca     = IncrementalPCA(n_components=8)
    n_skip   = 0

    t1 = time.time()
    for scene_dir in tqdm(scenes, desc='PCA pass'):
        sp = load_scene(scene_dir, args.max_size)
        if sp is None:
            n_skip += 1
            continue

        X_flat = (sp.astype(np.float64) / 10000.0).reshape(-1, 8)
        valid  = (X_flat > 0).all(axis=1)
        X_v    = X_flat[valid]
        if len(X_v) < 8:
            continue

        X_z = (X_v - mean) / std_safe   # global z-score
        ipca.partial_fit(X_z)

    elapsed2 = time.time() - t1
    print(f'\nPCA fit elapsed : {elapsed2 / 60:.1f} min')
    print(f'Skipped         : {n_skip}')
    print(f'\n{"PC":<5} {"Explained Var":>15}')
    print('-' * 22)
    for i, ev in enumerate(ipca.explained_variance_ratio_):
        print(f'PC{i+1:<3} {ev * 100:>14.2f}%')

    os.makedirs(os.path.dirname(os.path.abspath(args.pca_out)), exist_ok=True)
    np.savez(args.pca_out,
             components=ipca.components_,
             explained_variance_ratio=ipca.explained_variance_ratio_,
             mean=mean, std=std_safe)
    print(f'\nSaved → {args.pca_out}')


if __name__ == '__main__':
    main()
