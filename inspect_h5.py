"""
Zarr Patch Inspector & Visualizer for Landsat 8 Cloud Detection

사용법:
    # 데이터 내용 출력 (텍스트)
    python inspect_h5.py path/to/patch.zarr

    # 시각화 이미지 저장
    python inspect_h5.py path/to/patch.zarr --save

    # 여러 파일 랜덤 샘플링해서 저장
    python inspect_h5.py path/to/TRAIN_ZARR/ --sample 9 --save

예시:
    python inspect_h5.py data/TRAIN_ZARR/LC08_L1GT_160109_20201126_20210316_02_T2_PATCH0.zarr
    python inspect_h5.py data/TRAIN_ZARR/ --sample 6 --save --out output_vis/
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── 상수 ───────────────────────────────────────────────────────────────
BINARY_CLASS_NAMES  = {0: 'No-Cloud', 1: 'Cloud', 255: 'No-Data'}
BINARY_CLASS_COLORS = {
    0:   (0.13, 0.55, 0.13),  # Green  – No-Cloud
    1:   (1.0,  1.0,  1.0),   # White  – Cloud
    255: (0.0,  0.0,  0.0),   # Black  – No-Data
}


# ── 로더 ───────────────────────────────────────────────────────────────

def load_zarr(path: str) -> zarr.Group:
    return zarr.open_group(path, mode='r')


def print_zarr_info(path: str, store: zarr.Group):
    """텍스트로 zarr 패치 내용 요약 출력."""
    print('=' * 65)
    print(f'Patch : {os.path.basename(path)}')
    print(f'Arrays: {list(store.keys())}')
    print()

    for key in store.keys():
        arr = store[key][:]
        print(f'  [{key}]  shape={arr.shape}  dtype={arr.dtype}  '
              f'min={arr.min():.3f}  max={arr.max():.3f}  '
              f'mean={arr.mean():.3f}')

    # Label 분포
    for label_key in ('label', 'pseudo_label'):
        if label_key in store:
            lbl = store[label_key][:]
            total = lbl.size
            print(f'\n  [{label_key}] 클래스 분포:')
            for cls, name in sorted(BINARY_CLASS_NAMES.items()):
                cnt = int((lbl == cls).sum())
                pct = cnt / total * 100
                bar = '█' * int(pct / 2)
                print(f'    {cls:3d} {name:<10}: {cnt:>6} px ({pct:5.1f}%) {bar}')

    print('=' * 65)


# ── 유틸 함수 ──────────────────────────────────────────────────────────

def label_to_rgb(lbl: np.ndarray) -> np.ndarray:
    h, w = lbl.shape
    rgb  = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in BINARY_CLASS_COLORS.items():
        rgb[lbl == cls] = color
    return rgb


def make_legend_patches():
    return [mpatches.Patch(color=BINARY_CLASS_COLORS[c],
                           label=BINARY_CLASS_NAMES[c])
            for c in [0, 1, 255]]


# ── 시각화 ─────────────────────────────────────────────────────────────

def visualize_patch(path: str, save: bool = False, out_dir: str = '.'):
    store = load_zarr(path)
    print_zarr_info(path, store)

    spectral = store['spectral'][:].astype(np.float32) / 10000.0  # (H,W,8)
    rgb      = store['rgb'][:]     # (H,W,3) pre-normalised
    hsv      = store['hsv'][:]     # (H,W,3)
    sobel    = store['sobel'][:]   # (H,W,3)
    label    = store['label'][:]   # (H,W)

    has_pseudo = 'pseudo_label' in store

    n_rows = 3 if has_pseudo else 2
    n_cols = 4
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(os.path.basename(path), color='white', fontsize=10, y=1.01)

    def _ax(row, col):
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        return ax

    def show(ax, img, title, cmap=None, vmin=None, vmax=None, colorbar=False):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(title, color='#e0e0ff', fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')

    # Row 0: composites
    show(_ax(0, 0), rgb,              'True Color (RGB)')
    show(_ax(0, 1), hsv,              'HSV')
    sobel_mag = sobel[:, :, 2]
    vmax_s    = np.percentile(sobel_mag, 99) or 1.0
    show(_ax(0, 2), sobel_mag, 'Sobel Magnitude', cmap='hot', vmin=0, vmax=vmax_s)

    # NDSI from spectral bands
    g    = spectral[:, :, 2]
    s1   = spectral[:, :, 5]
    denom = g + s1
    denom[denom == 0] = 1e-6
    ndsi = (g - s1) / denom
    show(_ax(0, 3), ndsi, 'NDSI (spectral)', cmap='RdBu_r', vmin=-1, vmax=1,
         colorbar=True)

    # Row 1: individual bands
    band_pairs = [(0, 'B1 Coastal'), (4, 'B5 NIR'), (6, 'B7 SWIR2'), (7, 'B9 Cirrus')]
    for col, (bi, bname) in enumerate(band_pairs):
        band = spectral[:, :, bi]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        norm = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1) if hi > lo else band
        show(_ax(1, col), norm, bname, cmap='gray')

    # Row 2: labels + stats
    lbl_rgb = label_to_rgb(label)
    ax_lbl  = _ax(2 if has_pseudo else 1, 0)  # re-use slot
    ax_lbl  = _ax(n_rows - 1, 0)
    show(ax_lbl, lbl_rgb, 'QA_PIXEL (binary)')
    ax_lbl.legend(handles=make_legend_patches(), loc='lower right', fontsize=6,
                  facecolor='#0f0f23', labelcolor='white',
                  framealpha=0.7, handlelength=1.0)

    if has_pseudo:
        ps_lbl = store['pseudo_label'][:]
        show(_ax(n_rows - 1, 1), label_to_rgb(ps_lbl), 'Pseudo-label')

    # Pixel count bar chart
    ax_bar = _ax(n_rows - 1, 2)
    ax_bar.set_facecolor('#16213e')
    counts = [(label == c).sum() for c in [0, 1]]
    colors_bar = [BINARY_CLASS_COLORS[c] for c in [0, 1]]
    ax_bar.bar([0, 1], counts, color=colors_bar, edgecolor='#333355')
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(['No-Cloud', 'Cloud'], fontsize=8, color='#aaaaaa')
    ax_bar.set_title('Class Pixel Count', color='#e0e0ff', fontsize=9, pad=4)
    ax_bar.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#333355')

    # Spectral mean line
    ax_line = _ax(n_rows - 1, 3)
    ax_line.set_facecolor('#16213e')
    means = spectral.mean(axis=(0, 1))
    ax_line.plot(range(8), means, color='#7ec8e3', marker='o',
                 markersize=4, linewidth=1.5)
    ax_line.set_xticks(range(8))
    ax_line.set_xticklabels(['B1','B2','B3','B4','B5','B6','B7','B9'],
                             fontsize=7, color='#aaaaaa')
    ax_line.set_title('Mean Spectral (TOA refl.)', color='#e0e0ff', fontsize=9, pad=4)
    ax_line.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax_line.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout(pad=1.0)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(path))[0] + '_vis.png'
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {out_path}')
    else:
        plt.show()

    plt.close(fig)


def visualize_multiple(zarr_dir: str, n_samples: int, save: bool, out_dir: str):
    files = glob.glob(os.path.join(zarr_dir, '*.zarr'))
    if not files:
        print(f'.zarr 패치를 찾을 수 없습니다: {zarr_dir}')
        sys.exit(1)
    sampled = random.sample(files, min(n_samples, len(files)))
    print(f'\n{len(sampled)}개 패치 시각화 중...')
    for f in sampled:
        visualize_patch(f, save=save, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Landsat 8 Zarr 패치 내용 확인 및 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('path', help='.zarr 패치 경로 또는 ZARR 디렉토리 경로')
    parser.add_argument('--save', action='store_true',
                        help='화면 출력 대신 PNG 파일로 저장')
    parser.add_argument('--out', default='vis_output',
                        help='저장 디렉토리 (기본: vis_output)')
    parser.add_argument('--sample', type=int, default=1,
                        help='디렉토리 지정 시 랜덤 샘플 수 (기본: 1)')
    args = parser.parse_args()

    path = os.path.abspath(args.path)

    if os.path.isdir(path):
        # 단일 .zarr 패치인지, 상위 디렉토리인지 구분
        if os.path.exists(os.path.join(path, '.zgroup')):
            visualize_patch(path, save=args.save, out_dir=args.out)
        else:
            visualize_multiple(path, args.sample, args.save, args.out)
    else:
        print(f'경로가 없습니다: {path}')
        sys.exit(1)


if __name__ == '__main__':
    main()
