"""
vis_pca.py — PCA 8-component 3×3 grid + scatter + band-correlation for Landsat 8.

For each sampled scene:
  1. Fits PCA(n_components=8) on all 8 spectral bands (B1–B7, B9).
  2. Saves a 3×3 grid PNG: FCI (False Color Infrared) at (0,0), PC1–PC8 maps.
  3. Saves a 1×2 PC1 vs PC2 scatter PNG (density hexbin + luminance color).
  4. Saves an 8×8 Pearson-correlation heatmap (PCA component × spectral band).
  5. Saves correlation values as CSV.

Usage:
    conda run -n remote python vis_pca.py \\
        --root /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea \\
        --n 3 --out pca_vis/ --seed 42
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_cv_features import find_scenes, load_scene, pnorm

BAND_LABELS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9']
PC_LABELS   = [f'PC{i + 1}' for i in range(8)]


# ── PCA computation ────────────────────────────────────────────────────

def fit_pca(spectral: np.ndarray):
    """
    Fit PCA(8) on (H, W, 8) uint16 spectral array.

    Returns
    -------
    pca_model : fitted sklearn PCA object (reusable for transfer)
    pca_maps  : (H, W, 8) float32  — per-pixel component scores
    explained : (8,)      float64  — explained variance ratio per PC
    """
    H, W, _ = spectral.shape
    f = spectral.astype(np.float32) / 10000.0
    X = f.reshape(-1, 8)
    valid = np.isfinite(X).all(axis=1)

    pca = PCA(n_components=8, random_state=42)
    scores = np.zeros((H * W, 8), dtype=np.float32)
    if valid.sum() > 8:
        scores[valid] = pca.fit_transform(X[valid]).astype(np.float32)

    return pca, scores.reshape(H, W, 8), pca.explained_variance_ratio_


def compute_pca(spectral: np.ndarray):
    """Convenience wrapper — returns (pca_maps, explained) without the model."""
    _, maps, explained = fit_pca(spectral)
    return maps, explained


def compute_correlations(pca_maps: np.ndarray, spectral: np.ndarray) -> np.ndarray:
    """
    Pearson correlation between each PC and each spectral band.

    Returns
    -------
    corr : (8, 8) float32 — rows = PCs, cols = bands
    """
    f = spectral.astype(np.float32) / 10000.0
    X = f.reshape(-1, 8)
    S = pca_maps.reshape(-1, 8)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(S).all(axis=1)
    X_v, S_v = X[valid], S[valid]

    corr = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            if S_v[:, i].std() > 1e-9 and X_v[:, j].std() > 1e-9:
                corr[i, j] = float(np.corrcoef(S_v[:, i], X_v[:, j])[0, 1])
    return corr


# ── Plotting ───────────────────────────────────────────────────────────

def plot_pca_grid(spectral: np.ndarray, pca_maps: np.ndarray,
                  explained: np.ndarray, scene_id: str, out_path: str) -> None:
    """3×3 grid: FCI at (0,0), PC1–PC8 in the remaining 8 panels."""
    f = spectral.astype(np.float32) / 10000.0
    # FCI: NIR(B5) / Red(B4) / Green(B3)
    fci = np.stack([pnorm(f[:, :, 4]),
                    pnorm(f[:, :, 3]),
                    pnorm(f[:, :, 2])], axis=-1)

    fig, axes = plt.subplots(3, 3,
                             figsize=(3.2 * 3, 3.2 * 3),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.15})
    fig.suptitle(scene_id, fontsize=11, fontweight='bold', y=1.002)

    # (0, 0): FCI
    axes[0][0].imshow(np.clip(fci, 0, 1))
    axes[0][0].set_title('FCI  (NIR/R/G)', fontsize=9, pad=3)
    axes[0][0].axis('off')

    # Remaining 8 slots: PC1 → PC8
    slots = [(r, c) for r in range(3) for c in range(3)][1:]
    for pc_idx, (r, c) in enumerate(slots):
        pc_map = pca_maps[:, :, pc_idx]
        vabs   = float(np.nanpercentile(np.abs(pc_map), 98))
        ax     = axes[r][c]
        im     = ax.imshow(pc_map, cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(f'PC{pc_idx + 1}  ({explained[pc_idx] * 100:.1f}%)',
                     fontsize=9, pad=3)
        ax.axis('off')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def plot_correlation_heatmap(corr: np.ndarray, scene_id: str,
                             out_path: str) -> None:
    """8×8 annotated heatmap — rows = PCs, cols = bands."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.85)

    ax.set_xticks(range(8));  ax.set_xticklabels(BAND_LABELS, fontsize=10)
    ax.set_yticks(range(8));  ax.set_yticklabels(PC_LABELS,   fontsize=10)
    ax.set_xlabel('Spectral Band', fontsize=11)
    ax.set_ylabel('PCA Component', fontsize=11)
    ax.set_title(f'PCA–Band Pearson Correlation\n{scene_id}', fontsize=11)

    for i in range(8):
        for j in range(8):
            v = corr[i, j]
            txt_color = 'white' if abs(v) > 0.65 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def plot_pca_scatter(spectral: np.ndarray, pca_maps: np.ndarray,
                     scene_id: str, out_path: str,
                     max_pixels: int = 200_000) -> None:
    """
    1×2 scatter of PC1 (x) vs PC2 (y) for every pixel.
      Left  : hexbin density (log scale)
      Right : scatter colored by luminance (bright=cloud/snow, dark=shadow)
    Subsampled to max_pixels for speed.
    """
    f   = spectral.astype(np.float32) / 10000.0
    lum = (0.2989 * f[:, :, 3] + 0.5870 * f[:, :, 2]
           + 0.1140 * f[:, :, 1])          # luminance from R/G/B

    pc1 = pca_maps[:, :, 0].ravel()
    pc2 = pca_maps[:, :, 1].ravel()
    lum_flat = lum.ravel()
    valid = np.isfinite(pc1) & np.isfinite(pc2) & np.isfinite(lum_flat)
    pc1_v, pc2_v, lum_v = pc1[valid], pc2[valid], lum_flat[valid]

    if len(pc1_v) > max_pixels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pc1_v), max_pixels, replace=False)
        pc1_v, pc2_v, lum_v = pc1_v[idx], pc2_v[idx], lum_v[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'PC1 vs PC2 — {scene_id}', fontsize=10, fontweight='bold')

    # Left: density hexbin
    hb = axes[0].hexbin(pc1_v, pc2_v, gridsize=100, cmap='viridis',
                        bins='log', mincnt=1)
    plt.colorbar(hb, ax=axes[0], label='log₁₀(pixel count)')
    axes[0].set_xlabel('PC1', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    axes[0].set_title('Density', fontsize=10)

    # Right: luminance-colored scatter (bright=cloud/snow, dark=shadow/water)
    sc = axes[1].scatter(pc1_v, pc2_v, c=lum_v, cmap='gray',
                         s=0.3, alpha=0.25, vmin=0, vmax=0.7,
                         rasterized=True)
    plt.colorbar(sc, ax=axes[1], label='Luminance  (bright=cloud/snow)')
    axes[1].set_xlabel('PC1', fontsize=11)
    axes[1].set_ylabel('PC2', fontsize=11)
    axes[1].set_title('Luminance color', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def save_correlation_csv(corr: np.ndarray, out_path: str) -> None:
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([''] + BAND_LABELS)
        for i, label in enumerate(PC_LABELS):
            w.writerow([label] + [f'{corr[i, j]:.4f}' for j in range(8)])
    print(f"  → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PCA 3×3 grid + band-correlation heatmap for Landsat 8')
    parser.add_argument('--root', required=True,
                        help='Root folder containing Landsat scene directories')
    parser.add_argument('--n',    type=int, default=3,
                        help='Number of scenes to sample (default: 3)')
    parser.add_argument('--out',  default='pca_vis/',
                        help='Output directory (default: pca_vis/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)

    print('Searching for scenes...')
    scenes = find_scenes(args.root)
    print(f'Found {len(scenes)} scenes.')
    if not scenes:
        print('No scenes found. Check --root path.')
        return

    n       = min(args.n, len(scenes))
    sampled = random.sample(scenes, n)
    print(f'Sampled {n} scenes (seed={args.seed}):\n'
          + '\n'.join(f'  {Path(s).name}' for s in sampled))

    os.makedirs(args.out, exist_ok=True)

    for scene_dir in sampled:
        sid = Path(scene_dir).name
        print(f'\n[{sid}]')
        try:
            print('  Loading bands...')
            spectral = load_scene(scene_dir)
            print(f'  Shape: {spectral.shape}')

            print('  Computing PCA(8)...')
            pca_maps, explained = compute_pca(spectral)
            total_var = explained.sum() * 100
            print(f'  Explained variance: '
                  + ', '.join(f'PC{i+1}={v*100:.1f}%'
                               for i, v in enumerate(explained))
                  + f'  [total={total_var:.1f}%]')

            print('  Plotting 3×3 grid...')
            plot_pca_grid(
                spectral, pca_maps, explained, sid,
                os.path.join(args.out, f'{sid}_pca_grid.png'))

            print('  Plotting PC1 vs PC2 scatter...')
            plot_pca_scatter(
                spectral, pca_maps, sid,
                os.path.join(args.out, f'{sid}_pc1_pc2.png'))

            print('  Computing band correlations...')
            corr = compute_correlations(pca_maps, spectral)

            print('  Plotting correlation heatmap...')
            plot_correlation_heatmap(
                corr, sid,
                os.path.join(args.out, f'{sid}_pca_corr.png'))

            save_correlation_csv(
                corr,
                os.path.join(args.out, f'{sid}_pca_corr.csv'))

        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback; traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
