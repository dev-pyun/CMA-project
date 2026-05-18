"""
vis_pca_transfer.py — Cross-scene PCA transfer analysis for Landsat 8.

Fits PCA on scene A, then applies the same eigenvectors to scene B.
This tests whether PCA components are scene-invariant (good for use as
a fixed feature extractor) or scene-specific (needs per-scene fitting).

Outputs (saved to --out):
  {tag}_grid.png          2-row × 5-col: FCI + PC1-4 for A (fit) and B (transfer)
                          Both rows share the same colormap scale (from A's 98th pct)
  {tag}_scatter.png       PC1 vs PC2: overlay + hexbin A + hexbin B
  {sid_a}_corr.png/csv    Band-correlation for scene A
  {sid_b}_corr_transfer.png/csv  Band-correlation for scene B (A's PCA)

Usage:
    conda run -n remote python vis_pca_transfer.py \\
        --scene_a /earth00_home/.../LC08_L1GT_215108_20201213_20210314_02_T2 \\
        --scene_b /earth00_home/.../LC08_L1GT_201110_20241206_20241210_02_T2 \\
        --out pca_vis/

Interpreting the result:
  - If PC maps of B look structurally similar to A at the same scale
    → PCA axes transfer well → usable as a scene-agnostic feature
  - If the scatter distributions heavily overlap
    → PC1/PC2 cluster positions are consistent across scenes
  - If scales differ drastically → per-scene PCA normalisation needed
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_cv_features import load_scene, pnorm
from vis_pca import (
    fit_pca, load_global_stats, compute_correlations,
    plot_correlation_heatmap, save_correlation_csv,
)


# ── PCA transfer ───────────────────────────────────────────────────────

def apply_pca(pca_model: PCA, spectral: np.ndarray,
              scaler: dict | None = None) -> np.ndarray:
    """
    Apply a pre-fitted PCA (from scene A) to a new scene B.
    If scaler is provided (fitted on A), the same z-score normalization
    is applied to B before transforming — required for standardized PCA transfer.

    Returns
    -------
    pca_maps : (H, W, 8) float32 — scores in A's principal-component space
    """
    H, W, _ = spectral.shape
    f = spectral.astype(np.float32) / 10000.0
    X = f.reshape(-1, 8)
    valid = np.isfinite(X).all(axis=1)

    X_transform = X[valid].copy()
    if scaler is not None:
        X_transform = (X_transform - scaler['mean']) / scaler['std']

    scores = np.zeros((H * W, 8), dtype=np.float32)
    if valid.sum() > 8:
        scores[valid] = pca_model.transform(X_transform).astype(np.float32)
    return scores.reshape(H, W, 8)


# ── Plotting ───────────────────────────────────────────────────────────

def plot_transfer_grid(spectral_a: np.ndarray, spectral_b: np.ndarray,
                       maps_a: np.ndarray, maps_b: np.ndarray,
                       explained_a: np.ndarray,
                       sid_a: str, sid_b: str, out_path: str) -> None:
    """
    2-row × 5-col comparison grid.
      Row 0: FCI_A  + PC1-4 for scene A  (fit)
      Row 1: FCI_B  + PC1-4 for scene B  (A's PCA transferred)
    PC maps in both rows share the same vmin/vmax from A's 98th percentile,
    so direct visual comparison is meaningful.
    """
    def make_fci(sp: np.ndarray) -> np.ndarray:
        f = sp.astype(np.float32) / 10000.0
        return np.stack([pnorm(f[:, :, 4]),   # NIR
                         pnorm(f[:, :, 3]),   # Red
                         pnorm(f[:, :, 2])],  # Green
                        axis=-1)

    fci_a, fci_b = make_fci(spectral_a), make_fci(spectral_b)

    # Common scale: 98th-percentile absolute value from scene A
    vabs = [float(np.nanpercentile(np.abs(maps_a[:, :, i]), 98))
            for i in range(4)]

    fig, axes = plt.subplots(
        2, 5, figsize=(5 * 3.5, 2 * 3.5),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.18})
    fig.suptitle(
        f'PCA Transfer\nA (fit): {Path(sid_a).name[:45]}\n'
        f'B (transfer): {Path(sid_b).name[:45]}',
        fontsize=8, fontweight='bold', y=1.03)

    for row, (fci, maps, label) in enumerate([
            (fci_a, maps_a, 'A  [fit]'),
            (fci_b, maps_b, 'B  [A→B transfer]')]):
        # FCI panel
        axes[row][0].imshow(np.clip(fci, 0, 1))
        axes[row][0].set_title(f'FCI  {label}', fontsize=8, pad=3)
        axes[row][0].axis('off')

        # PC1–PC4 panels
        for pc_idx in range(4):
            ax  = axes[row][pc_idx + 1]
            im  = ax.imshow(maps[:, :, pc_idx], cmap='RdBu_r',
                            vmin=-vabs[pc_idx], vmax=vabs[pc_idx],
                            interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            title = f'PC{pc_idx + 1}'
            if row == 0:
                title += f'  ({explained_a[pc_idx] * 100:.1f}%)'
            ax.set_title(title, fontsize=8, pad=3)
            ax.axis('off')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def plot_transfer_scatter(maps_a: np.ndarray, maps_b: np.ndarray,
                          sid_a: str, sid_b: str, out_path: str,
                          max_pixels: int = 200_000) -> None:
    """
    1×3 scatter figure comparing PC1 vs PC2 distributions of A and B.
      Left   : overlay (A=blue, B=orange) — shape of distributions
      Centre : hexbin density for scene A
      Right  : hexbin density for scene B  (A's coordinate system)
    Overlapping distributions → PCA transfer is geometrically consistent.
    """
    def _sample(maps: np.ndarray, n: int):
        pc1, pc2 = maps[:, :, 0].ravel(), maps[:, :, 1].ravel()
        ok = np.isfinite(pc1) & np.isfinite(pc2)
        pc1, pc2 = pc1[ok], pc2[ok]
        if len(pc1) > n:
            idx = np.random.default_rng(42).choice(len(pc1), n, replace=False)
            pc1, pc2 = pc1[idx], pc2[idx]
        return pc1, pc2

    pc1_a, pc2_a = _sample(maps_a, max_pixels)
    pc1_b, pc2_b = _sample(maps_b, max_pixels)

    # Shared axis limits (union of both distributions)
    all_pc1 = np.concatenate([pc1_a, pc1_b])
    all_pc2 = np.concatenate([pc2_a, pc2_b])
    xlim = (np.percentile(all_pc1, 1), np.percentile(all_pc1, 99))
    ylim = (np.percentile(all_pc2, 1), np.percentile(all_pc2, 99))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('PC1 vs PC2 — Transfer Comparison  (A\'s eigenvectors)',
                 fontsize=10, fontweight='bold')

    # Left: overlay scatter (subsampled for clarity)
    n_ov = min(60_000, len(pc1_a), len(pc1_b))
    axes[0].scatter(pc1_a[:n_ov], pc2_a[:n_ov],
                    s=0.4, alpha=0.2, c='steelblue',
                    label=f'A: {Path(sid_a).name[:22]}', rasterized=True)
    axes[0].scatter(pc1_b[:n_ov], pc2_b[:n_ov],
                    s=0.4, alpha=0.2, c='tomato',
                    label=f'B: {Path(sid_b).name[:22]}', rasterized=True)
    axes[0].set_xlim(xlim); axes[0].set_ylim(ylim)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Overlay  (A=blue, B=orange)', fontsize=9)
    axes[0].legend(fontsize=7, markerscale=10)

    # Centre: hexbin scene A
    hb_a = axes[1].hexbin(pc1_a, pc2_a, gridsize=80, cmap='Blues',
                          bins='log', mincnt=1, extent=(*xlim, *ylim))
    plt.colorbar(hb_a, ax=axes[1], label='log₁₀(count)')
    axes[1].set_xlim(xlim); axes[1].set_ylim(ylim)
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title(f'A  [fit]\n{Path(sid_a).name[:35]}', fontsize=8)

    # Right: hexbin scene B (same axes)
    hb_b = axes[2].hexbin(pc1_b, pc2_b, gridsize=80, cmap='Oranges',
                          bins='log', mincnt=1, extent=(*xlim, *ylim))
    plt.colorbar(hb_b, ax=axes[2], label='log₁₀(count)')
    axes[2].set_xlim(xlim); axes[2].set_ylim(ylim)
    axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
    axes[2].set_title(f'B  [A→B transfer]\n{Path(sid_b).name[:35]}', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cross-scene PCA transfer analysis for Landsat 8')
    parser.add_argument('--scene_a', required=True,
                        help='Source scene directory (PCA fitted here)')
    parser.add_argument('--scene_b', required=True,
                        help='Target scene directory (PCA transferred here)')
    parser.add_argument('--out', default='pca_vis/',
                        help='Output directory (default: pca_vis/)')
    parser.add_argument('--global_stats', default=None,
                        help='Path to global_spectral_stats.npz for standardized PCA')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sid_a = Path(args.scene_a).name
    sid_b = Path(args.scene_b).name
    tag   = f'{sid_a[:28]}_to_{sid_b[:28]}'

    # ── Load and fit scene A ───────────────────────────────────────────
    print(f'[Scene A — fit]  {sid_a}')
    print('  Loading...')
    sp_a = load_scene(args.scene_a)
    print(f'  Shape: {sp_a.shape}')
    print('  Fitting PCA(8)...')
    gstats = load_global_stats(args.global_stats) if args.global_stats else None
    std_flag = gstats is not None

    pca_model, maps_a, explained_a, scaler_a = fit_pca(
        sp_a, standardize=std_flag, global_stats=gstats)
    std_label = ' [global-std]' if gstats else ''
    print(f'  Explained variance{std_label}: '
          + ', '.join(f'PC{i+1}={v*100:.1f}%' for i, v in enumerate(explained_a))
          + f'  [total={explained_a.sum()*100:.1f}%]')

    # ── Load scene B and apply A's PCA ────────────────────────────────
    print(f'\n[Scene B — transfer]  {sid_b}')
    print('  Loading...')
    sp_b = load_scene(args.scene_b)
    print(f'  Shape: {sp_b.shape}')
    print("  Applying A's PCA to B...")
    maps_b = apply_pca(pca_model, sp_b, scaler=scaler_a)

    # ── Plots ─────────────────────────────────────────────────────────
    print('\nPlotting PC1-4 transfer grid...')
    plot_transfer_grid(sp_a, sp_b, maps_a, maps_b, explained_a,
                       sid_a, sid_b,
                       os.path.join(args.out, f'{tag}_grid.png'))

    print('Plotting PC1 vs PC2 scatter comparison...')
    plot_transfer_scatter(maps_a, maps_b, sid_a, sid_b,
                          os.path.join(args.out, f'{tag}_scatter.png'))

    # ── Correlations ──────────────────────────────────────────────────
    print('Computing band correlations...')
    corr_a = compute_correlations(maps_a, sp_a)
    corr_b = compute_correlations(maps_b, sp_b)

    plot_correlation_heatmap(corr_a, f'{sid_a}  [fit]',
                             os.path.join(args.out, f'{sid_a[:30]}_corr.png'))
    plot_correlation_heatmap(corr_b, f'{sid_b}  [A→B transfer]',
                             os.path.join(args.out,
                                          f'{sid_b[:30]}_corr_transfer.png'))
    save_correlation_csv(corr_a,
                         os.path.join(args.out, f'{sid_a[:30]}_corr.csv'))
    save_correlation_csv(corr_b,
                         os.path.join(args.out,
                                      f'{sid_b[:30]}_corr_transfer.csv'))

    print('\nDone.')


if __name__ == '__main__':
    main()
