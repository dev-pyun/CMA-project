"""
vis_cv_features.py — 7×6 CV feature grid for Landsat 8 scenes.

Randomly samples N scenes from a root folder and generates a 7×6
grid image of CV features for each scene to aid input channel selection.

Usage:
    conda run -n remote python vis_cv_features.py \\
        --root /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea \\
        --n 5 --out cv_vis/ --seed 42

Grid layout (7×6 = 42 panels):
  Row 1  Raw Spectral      RGB / B1 / B5 NIR / B6 SWIR1 / B7 SWIR2 / B9 Cirrus
  Row 2  Spectral Index    NDSI / NDWI / NDVI / MNDWI / Gray-world RGB / Brightness
  Row 3  Color Space       H / S / V / Entropy(H) / Entropy(S) / Entropy(V)
  Row 4  Edge/Gradient     Sobel Mag / Sobel X / Sobel Y / Laplacian / DoG / Canny
  Row 5  Texture           Local Entropy / Local StdDev / LBP / White Top-hat / Local CoV / Local Range
  Row 6  Spectral Trans.   HOT / Vis Brightness / SWIR Ratio / PCA1 / PCA2 / PCA3
  Row 7  Lab / FFT         Lab a / Lab b / FFT Mag (GFD) / FFT Low-freq / FFT High-freq / GFD Edge FFT
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import (
    laplace, sobel as ndimage_sobel,
    uniform_filter, maximum_filter, minimum_filter,
)
from skimage.feature import local_binary_pattern, canny
from skimage.filters import gaussian
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, white_tophat
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.split_scene import (
    ALL_BANDS, N_SPECTRAL,
    find_band_file,
    _dn_to_toa_uint16, _load_sun_sin,
)

MAX_SIZE = 1000   # max pixels on longest side for display & computation
WINDOW   = 9      # local filter window (pixels)
ROW_LABELS = [
    'Raw Spectral',
    'Spectral Index + Color',
    'Color Space + Entropy',
    'Edge / Gradient',
    'Texture',
    'Spectral Transform',
    'Lab / FFT',
]


# ── Scene discovery ────────────────────────────────────────────────────

def find_scenes(root: str) -> list[str]:
    """Recursively find Landsat scene directories (contain *_B1.TIF)."""
    scenes = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.endswith('_B1.TIF') for f in filenames):
            scenes.append(dirpath)
    return sorted(scenes)


# ── Data loading ───────────────────────────────────────────────────────

def load_scene(scene_dir: str) -> np.ndarray:
    """
    Load TOA-converted spectral bands, downsampled to MAX_SIZE.
    Returns (H, W, 8) uint16 (×10000 reflectance scale).
    """
    band_files = {}
    for bk in ALL_BANDS:
        bf = find_band_file(scene_dir, bk)
        if bf:
            band_files[bk] = bf
        elif bk != 'B9':
            raise FileNotFoundError(f"Missing band {bk} in {scene_dir}")

    with rasterio.open(list(band_files.values())[0]) as src:
        H, W = src.height, src.width

    scale = max(1, max(H, W) // MAX_SIZE)

    spectral = np.zeros((H, W, N_SPECTRAL), dtype=np.uint16)
    for ch, bk in enumerate(ALL_BANDS):
        if bk not in band_files:
            continue
        with rasterio.open(band_files[bk]) as src:
            spectral[:, :, ch] = src.read(1)

    sun_sin  = _load_sun_sin(scene_dir)
    spectral = _dn_to_toa_uint16(spectral, sun_sin=sun_sin)

    return spectral[::scale, ::scale]


# ── Helpers ────────────────────────────────────────────────────────────

def pnorm(arr: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    """Percentile-stretch to [0, 1], ignoring NaN/Inf."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1).astype(np.float32)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ratio clipped to [-1, 1] with zero-denominator guard."""
    d = b.copy().astype(np.float32)
    d[np.abs(d) < 1e-6] = 1e-6
    return np.clip(a / d, -1, 1).astype(np.float32)


def local_std_fast(arr: np.ndarray, size: int) -> np.ndarray:
    mean    = uniform_filter(arr.astype(np.float64), size=size)
    mean_sq = uniform_filter((arr.astype(np.float64)) ** 2, size=size)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0)).astype(np.float32)


def local_range_fast(arr: np.ndarray, size: int) -> np.ndarray:
    return (maximum_filter(arr, size=size)
            - minimum_filter(arr, size=size)).astype(np.float32)


def local_cov_fast(arr: np.ndarray, size: int) -> np.ndarray:
    mean = uniform_filter(arr.astype(np.float64), size=size)
    std  = local_std_fast(arr, size)
    cov  = np.where(mean > 1e-6, std / (mean + 1e-6), 0.0)
    return cov.astype(np.float32)


# ── Feature computation ────────────────────────────────────────────────

def compute_features(spectral: np.ndarray) -> list[tuple]:
    """
    Compute all 36 CV features.
    Returns list of (title, array, cmap, vmin, vmax).
    RGB panels have cmap=None and array shape (H, W, 3).
    """
    f = spectral.astype(np.float32) / 10000.0   # reflectance in [0, ~1]

    b1 = f[:, :, 0]   # Coastal/Aerosol
    b2 = f[:, :, 1]   # Blue
    b3 = f[:, :, 2]   # Green
    b4 = f[:, :, 3]   # Red
    b5 = f[:, :, 4]   # NIR
    b6 = f[:, :, 5]   # SWIR1
    b7 = f[:, :, 6]   # SWIR2
    b9 = f[:, :, 7]   # Cirrus

    # ── RGB composite ─────────────────────────────────────────────────
    rgb = np.stack([pnorm(b4), pnorm(b3), pnorm(b2)], axis=-1)

    # ── HSV ───────────────────────────────────────────────────────────
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    H_ch   = hsv[:, :, 0].astype(np.float32) / 179.0
    S_ch   = hsv[:, :, 1].astype(np.float32) / 255.0
    V_ch   = hsv[:, :, 2].astype(np.float32) / 255.0

    # ── Luminance for gradient/texture ───────────────────────────────
    lum    = (0.2989 * rgb[:, :, 0]
              + 0.5870 * rgb[:, :, 1]
              + 0.1140 * rgb[:, :, 2]).astype(np.float32)
    lum_u8 = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    H_u8   = (np.clip(H_ch, 0, 1) * 255).astype(np.uint8)
    S_u8   = (np.clip(S_ch, 0, 1) * 255).astype(np.uint8)

    # ── Sobel (scipy, axis=1→x, axis=0→y) ────────────────────────────
    lum64 = lum.astype(np.float64)
    sx    = ndimage_sobel(lum64, axis=1).astype(np.float32)
    sy    = ndimage_sobel(lum64, axis=0).astype(np.float32)
    smag  = np.sqrt(sx ** 2 + sy ** 2)

    # ── Laplacian / DoG / Canny ───────────────────────────────────────
    lap = laplace(lum64).astype(np.float32)
    dog = (gaussian(lum, sigma=1) - gaussian(lum, sigma=3)).astype(np.float32)
    can = canny(lum, sigma=1).astype(np.float32)

    # ── Local texture ─────────────────────────────────────────────────
    r       = disk(WINDOW // 2)
    loc_ent = rank_entropy(lum_u8, r).astype(np.float32)
    ent_h   = rank_entropy(H_u8,   r).astype(np.float32)
    ent_s   = rank_entropy(S_u8,   r).astype(np.float32)
    ent_v   = rank_entropy(lum_u8, r).astype(np.float32)  # V ≈ luminance

    l_std   = local_std_fast(lum,   WINDOW)
    l_range = local_range_fast(lum, WINDOW)
    l_cov   = local_cov_fast(lum,   WINDOW)

    lbp = local_binary_pattern(
        lum_u8, P=8, R=1, method='uniform').astype(np.float32)
    wth = white_tophat(lum_u8, disk(7)).astype(np.float32)

    # ── Spectral indices ──────────────────────────────────────────────
    ndsi  = safe_div(b5 - b6, b5 + b6)
    ndwi  = safe_div(b3 - b5, b3 + b5)
    ndvi  = safe_div(b5 - b4, b5 + b4)
    mndwi = safe_div(b3 - b6, b3 + b6)
    brt_s = (b2 + b3 + b4) / 3.0

    # ── Gray-world color correction ───────────────────────────────────
    # Each channel normalised so its mean equals 0.5 (neutral grey)
    rgb_gw = rgb.copy().astype(np.float32)
    for c in range(3):
        ch_mean = rgb_gw[:, :, c].mean()
        if ch_mean > 1e-6:
            rgb_gw[:, :, c] = np.clip(rgb_gw[:, :, c] * 0.5 / ch_mean, 0, 1)

    # ── Lab a / b channels ────────────────────────────────────────────
    # cv2 maps L→0~255, a/b: -128~127 → 0~255
    lab     = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lab_a   = (lab[:, :, 1].astype(np.float32) - 128.0) / 127.0   # [-1, 1]
    lab_b   = (lab[:, :, 2].astype(np.float32) - 128.0) / 127.0   # [-1, 1]

    # ── FFT features (luminance) ──────────────────────────────────────
    fft2d        = np.fft.fft2(lum.astype(np.float64))
    fft_shifted  = np.fft.fftshift(fft2d)
    fft_mag      = np.log1p(np.abs(fft_shifted)).astype(np.float32)

    H_f, W_f = lum.shape
    cy, cx   = H_f // 2, W_f // 2
    r_low    = min(H_f, W_f) // 8          # radius covering lowest ~1% of freqs
    yy, xx   = np.ogrid[:H_f, :W_f]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_low ** 2

    fft_lo       = fft_shifted.copy();  fft_lo[~mask_low] = 0
    fft_hi       = fft_shifted.copy();  fft_hi[mask_low]  = 0
    fft_low_rec  = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_lo))).astype(np.float32)
    fft_high_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_hi))).astype(np.float32)

    # ── GFD proxy: FFT of Canny edge map ──────────────────────────────
    fft_edge    = np.fft.fftshift(np.fft.fft2(can.astype(np.float64)))
    gfd_edge    = np.log1p(np.abs(fft_edge)).astype(np.float32)

    # ── Row 6: spectral transforms ────────────────────────────────────
    hot        = (b1 - 0.5 * b3).astype(np.float32)   # Haze Optimized Transform
    swir_ratio = safe_div(b6, b7)

    H_px, W_px = spectral.shape[:2]
    X = f.reshape(-1, 8)
    pca_map = np.zeros((H_px * W_px, 3), dtype=np.float32)
    valid   = np.isfinite(X).all(axis=1)
    if valid.sum() > 3:
        pca = PCA(n_components=3)
        pca_map[valid] = pca.fit_transform(X[valid]).astype(np.float32)
    pc1 = pca_map[:, 0].reshape(H_px, W_px)
    pc2 = pca_map[:, 1].reshape(H_px, W_px)
    pc3 = pca_map[:, 2].reshape(H_px, W_px)

    # ── Assemble (title, data, cmap, vmin, vmax) ──────────────────────
    DIV  = 'RdBu_r'
    SEQ  = 'viridis'
    HOT  = 'hot'
    GRAY = 'gray'

    return [
        # Row 1: Raw Spectral
        ('RGB',           rgb,          None, None, None),
        ('B1 Coastal',    pnorm(b1),    SEQ,  0,    1),
        ('B5 NIR',        pnorm(b5),    SEQ,  0,    1),
        ('B6 SWIR1',      pnorm(b6),    SEQ,  0,    1),
        ('B7 SWIR2',      pnorm(b7),    SEQ,  0,    1),
        ('B9 Cirrus',     pnorm(b9),    SEQ,  0,    1),
        # Row 2: Spectral Index + Color Correction
        ('NDSI',          ndsi,         DIV,  -1,   1),
        ('NDWI',          ndwi,         DIV,  -1,   1),
        ('NDVI',          ndvi,         DIV,  -1,   1),
        ('MNDWI',         mndwi,        DIV,  -1,   1),
        ('Gray-world RGB',rgb_gw,       None, None, None),
        ('Brightness',    pnorm(brt_s), SEQ,  0,    1),
        # Row 3: Color Space + Entropy
        ('Hue (H)',       H_ch,         'hsv', 0,   1),
        ('Saturation(S)', S_ch,         SEQ,  0,    1),
        ('Value (V)',     V_ch,         SEQ,  0,    1),
        ('Entropy(H)',    ent_h,        HOT,  None, None),
        ('Entropy(S)',    ent_s,        HOT,  None, None),
        ('Entropy(V)',    ent_v,        HOT,  None, None),
        # Row 4: Edge / Gradient
        ('Sobel Mag',     smag,         SEQ,  None, None),
        ('Sobel X',       sx,           DIV,  None, None),
        ('Sobel Y',       sy,           DIV,  None, None),
        ('Laplacian',     lap,          DIV,  None, None),
        ('DoG',           dog,          DIV,  None, None),
        ('Canny',         can,          GRAY, 0,    1),
        # Row 5: Texture
        ('Local Entropy', loc_ent,      HOT,  None, None),
        ('Local StdDev',  l_std,        SEQ,  None, None),
        ('LBP',           lbp,          SEQ,  None, None),
        ('White Top-hat', wth,          SEQ,  None, None),
        ('Local CoV',     l_cov,        SEQ,  None, None),
        ('Local Range',   l_range,      SEQ,  None, None),
        # Row 6: Spectral Transforms
        ('HOT',           hot,          DIV,  None, None),
        ('Vis Bright',    pnorm(brt_s), SEQ,  0,    1),
        ('SWIR B6/B7',    swir_ratio,   SEQ,  0,    1),
        ('PCA 1',         pc1,          DIV,  None, None),
        ('PCA 2',         pc2,          DIV,  None, None),
        ('PCA 3',         pc3,          DIV,  None, None),
        # Row 7: Lab / FFT Analysis
        ('Lab a (G–R)',   lab_a,        DIV,  -1,   1),
        ('Lab b (B–Y)',   lab_b,        DIV,  -1,   1),
        ('FFT Mag (GFD)', fft_mag,      HOT,  None, None),
        ('FFT Low-freq',  fft_low_rec,  SEQ,  None, None),
        ('FFT High-freq', fft_high_rec, SEQ,  None, None),
        ('GFD Edge FFT',  gfd_edge,     HOT,  None, None),
    ]


# ── Plotting ───────────────────────────────────────────────────────────

def plot_grid(features: list, scene_id: str, out_path: str) -> None:
    ROWS, COLS = 7, 6
    fig, axes = plt.subplots(
        ROWS, COLS,
        figsize=(COLS * 3.2, ROWS * 3.2),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.05},
    )
    fig.suptitle(scene_id, fontsize=13, fontweight='bold', y=1.002)

    for idx, (title, data, cmap, vmin, vmax) in enumerate(features):
        row, col = divmod(idx, COLS)
        ax = axes[row][col]

        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax.set_title(title, fontsize=8, pad=3)
        ax.axis('off')

        # Row label on first column
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row], fontsize=8,
                          rotation=90, labelpad=6, va='center')
            ax.yaxis.set_visible(True)
            ax.tick_params(left=False, labelleft=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='7×6 CV feature grid for Landsat 8 scenes')
    parser.add_argument('--root',  required=True,
                        help='Root folder containing Landsat scene directories')
    parser.add_argument('--n',     type=int, default=5,
                        help='Number of scenes to randomly sample (default: 5)')
    parser.add_argument('--out',   default='cv_vis/',
                        help='Output directory (default: cv_vis/)')
    parser.add_argument('--seed',  type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)

    print('Searching for scenes...')
    scenes = find_scenes(args.root)
    print(f'Found {len(scenes)} scenes.')

    if not scenes:
        print('No scenes found. Check --root path.')
        return

    n = min(args.n, len(scenes))
    sampled = random.sample(scenes, n)
    print(f'Sampled {n} scenes (seed={args.seed}):\n'
          + '\n'.join(f'  {Path(s).name}' for s in sampled))

    os.makedirs(args.out, exist_ok=True)

    for scene_dir in sampled:
        scene_id = Path(scene_dir).name
        out_path = os.path.join(args.out, f'{scene_id}.png')
        print(f'\n[{scene_id}]')
        try:
            print('  Loading bands...')
            spectral = load_scene(scene_dir)
            print(f'  Shape after downsample: {spectral.shape}')
            print('  Computing features...')
            features = compute_features(spectral)
            print('  Plotting...')
            plot_grid(features, scene_id, out_path)
        except Exception as e:
            print(f'  ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
