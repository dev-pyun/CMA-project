"""
Zarr Patch Inspector & Visualizer for Landsat 8 Cloud Detection

한때 h5를 썼다보니... 이름이 inspect_h5였지만! 바꿈!!

사용법:
    # 데이터 내용 출력 (텍스트)
    python inspect_zarr.py path/to/patch.zarr

    # 시각화 이미지 저장
    python inspect_zarr.py path/to/patch.zarr --save

    # 여러 파일 랜덤 샘플링해서 저장
    python inspect_zarr.py path/to/TRAIN_ZARR/ --sample 9 --save

예시:
    python inspect_zarr.py data/TRAIN_ZARR/LC08_L1GT_160109_20201126_20210316_02_T2_PATCH0.zarr
    python inspect_zarr.py data/TRAIN_ZARR/ --sample 6 --save --out output_vis/
"""

import argparse
import glob
import os
import random
import re
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


# ── 패치 위치 계산 ─────────────────────────────────────────────────────

PATCH_SIZE = 256  # split_scene.py 기본값
PATCH_OVERLAP = 0

# FCI 기본 탐색 경로 (스크립트 위치 기준 label_code/prepared/)
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FCI_DIR = os.path.join(_SCRIPT_DIR, 'label_code', 'prepared')


def parse_patch_name(patch_path: str):
    """'..._{scene_id}_PATCH{n}.zarr' → (scene_id, patch_idx) 또는 (None, None)."""
    basename = os.path.splitext(os.path.basename(patch_path))[0]
    m = re.match(r'^(.+)_PATCH(\d+)$', basename)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def find_fci(scene_id: str, fci_dir: str) -> str | None:
    """fci_dir 아래에서 scene_id 에 해당하는 fci.tif 경로를 찾아 반환."""
    candidate = os.path.join(fci_dir, scene_id, 'fci.tif')
    return candidate if os.path.exists(candidate) else None


def compute_patch_bbox(patch_idx: int, img_h: int, img_w: int,
                       patch_size: int = PATCH_SIZE,
                       overlap: int = PATCH_OVERLAP):
    """패치 인덱스 → 씬 내 위치 (row_start, col_start) 픽셀 좌표."""
    step = patch_size - overlap
    n_patches_x = max(1, (img_w - overlap) // step)
    iy = patch_idx // n_patches_x
    ix = patch_idx % n_patches_x
    row_start = min(iy * step, max(0, img_h - patch_size))
    col_start = min(ix * step, max(0, img_w - patch_size))
    return row_start, col_start


def make_scene_overview(fci_path: str, row_start: int, col_start: int,
                        patch_size: int = PATCH_SIZE,
                        max_side: int = 512) -> np.ndarray | None:
    """
    FCI 썸네일 위에 패치 위치를 빨간 박스로 표시한 RGB 배열 반환.
    max_side: 긴 변의 최대 픽셀 수 (다운샘플링 기준).
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        return None

    with rasterio.open(fci_path) as src:
        H, W = src.height, src.width
        scale = min(max_side / H, max_side / W, 1.0)
        out_h, out_w = max(1, int(H * scale)), max(1, int(W * scale))
        fci = src.read(out_shape=(3, out_h, out_w), resampling=Resampling.average)

    thumb = np.transpose(fci, (1, 2, 0)).copy()   # (out_h, out_w, 3) uint8

    # 패치 박스 좌표 (스케일 적용)
    r0 = int(row_start * scale)
    c0 = int(col_start * scale)
    r1 = min(out_h - 1, r0 + max(1, int(patch_size * scale)))
    c1 = min(out_w - 1, c0 + max(1, int(patch_size * scale)))

    # 빨간 테두리 (3px)
    t = 3
    thumb[r0:r0+t, c0:c1] = [255, 50, 50]
    thumb[r1-t:r1, c0:c1] = [255, 50, 50]
    thumb[r0:r1, c0:c0+t] = [255, 50, 50]
    thumb[r0:r1, c1-t:c1] = [255, 50, 50]

    return thumb


# ── 시각화 ─────────────────────────────────────────────────────────────

def visualize_patch(path: str, save: bool = False, out_dir: str = '.',
                    fci_dir: str = DEFAULT_FCI_DIR):
    store = load_zarr(path)
    print_zarr_info(path, store)

    spectral = store['spectral'][:].astype(np.float32) / 10000.0  # (H,W,8)
    rgb      = store['rgb'][:]     # (H,W,3) pre-normalised
    hsv      = store['hsv'][:]     # (H,W,3)
    sobel    = store['sobel'][:]   # (H,W,3)
    label    = store['label'][:]   # (H,W)

    has_pseudo = 'pseudo_label' in store

    # ── 씬 오버뷰 (fci_dir 에서 FCI 자동 탐색, 없으면 조용히 스킵) ────
    overview       = None
    overview_title = ''
    scene_id, patch_idx = parse_patch_name(path)
    if scene_id is not None:
        fci_path = find_fci(scene_id, fci_dir)
        if fci_path:
            try:
                import rasterio as _rio
                with _rio.open(fci_path) as src:
                    img_h, img_w = src.height, src.width
                row_start, col_start = compute_patch_bbox(patch_idx, img_h, img_w)
                overview = make_scene_overview(fci_path, row_start, col_start)
                overview_title = (
                    f'Scene Overview ({img_h}×{img_w} px)  '
                    f'│  Patch #{patch_idx}  @row={row_start}, col={col_start}'
                )
                print(f'[overview] patch #{patch_idx} → row={row_start}, col={col_start}')
            except Exception as e:
                print(f'[overview] FCI 로드 실패: {e}')

    n_base = 3 if has_pseudo else 2
    has_ov = overview is not None
    n_rows = n_base + (1 if has_ov else 0)
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

    def _style(ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

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
    g     = spectral[:, :, 2]
    s1    = spectral[:, :, 5]
    denom = g + s1
    denom[denom == 0] = 1e-6
    ndsi  = (g - s1) / denom
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
    stat_row = 2 if has_pseudo else 1
    lbl_rgb  = label_to_rgb(label)
    ax_lbl   = _ax(stat_row, 0)
    show(ax_lbl, lbl_rgb, 'Label (binary)')
    ax_lbl.legend(handles=make_legend_patches(), loc='lower right', fontsize=6,
                  facecolor='#0f0f23', labelcolor='white',
                  framealpha=0.7, handlelength=1.0)

    if has_pseudo:
        ps_lbl = store['pseudo_label'][:]
        show(_ax(stat_row, 1), label_to_rgb(ps_lbl), 'Pseudo-label')

    # Pixel count bar chart
    ax_bar = _ax(stat_row, 2)
    _style(ax_bar)
    counts     = [(label == c).sum() for c in [0, 1]]
    colors_bar = [BINARY_CLASS_COLORS[c] for c in [0, 1]]
    ax_bar.bar([0, 1], counts, color=colors_bar, edgecolor='#333355')
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(['No-Cloud', 'Cloud'], fontsize=8, color='#aaaaaa')
    ax_bar.set_title('Class Pixel Count', color='#e0e0ff', fontsize=9, pad=4)

    # Spectral mean line
    ax_line = _ax(stat_row, 3)
    _style(ax_line)
    means = spectral.mean(axis=(0, 1))
    ax_line.plot(range(8), means, color='#7ec8e3', marker='o',
                 markersize=4, linewidth=1.5)
    ax_line.set_xticks(range(8))
    ax_line.set_xticklabels(['B1','B2','B3','B4','B5','B6','B7','B9'],
                             fontsize=7, color='#aaaaaa')
    ax_line.set_title('Mean Spectral (TOA refl.)', color='#e0e0ff', fontsize=9, pad=4)

    # ── 씬 오버뷰 행 (FCI 있을 때만) ────────────────────────────────────
    if has_ov:
        ov_row = stat_row + 1
        ax_ov  = plt.subplot2grid((n_rows, n_cols), (ov_row, 0), colspan=n_cols, fig=fig)
        _style(ax_ov)
        ax_ov.imshow(overview, interpolation='bilinear')
        ax_ov.set_title(overview_title, color='#ff8888', fontsize=9, pad=4)
        ax_ov.set_xticks([])
        ax_ov.set_yticks([])

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


def visualize_multiple(zarr_dir: str, n_samples: int, save: bool, out_dir: str,
                       fci_dir: str = DEFAULT_FCI_DIR):
    files = glob.glob(os.path.join(zarr_dir, '*.zarr'))
    if not files:
        print(f'.zarr 패치를 찾을 수 없습니다: {zarr_dir}')
        sys.exit(1)
    sampled = random.sample(files, min(n_samples, len(files)))
    print(f'\n{len(sampled)}개 패치 시각화 중...')
    for f in sampled:
        visualize_patch(f, save=save, out_dir=out_dir, fci_dir=fci_dir)


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
    parser.add_argument('--fci_dir', default=DEFAULT_FCI_DIR,
                        help=f'씬 FCI 탐색 루트 디렉토리 (기본: {DEFAULT_FCI_DIR})')
    args = parser.parse_args()

    path = os.path.abspath(args.path)

    if os.path.isdir(path):
        if (os.path.exists(os.path.join(path, '.zgroup')) or
                os.path.exists(os.path.join(path, 'zarr.json'))):
            visualize_patch(path, save=args.save, out_dir=args.out, fci_dir=args.fci_dir)
        else:
            visualize_multiple(path, args.sample, args.save, args.out, fci_dir=args.fci_dir)
    else:
        print(f'경로가 없습니다: {path}')
        sys.exit(1)


if __name__ == '__main__':
    main()
