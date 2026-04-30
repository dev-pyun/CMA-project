"""
H5 Patch Inspector & Visualizer for Landsat 8 Cloud/Shadow/Snow Detection

사용법:
    # 데이터 내용 출력 (텍스트)
    python inspect_h5.py path/to/patch.h5

    # 시각화 이미지 저장
    python inspect_h5.py path/to/patch.h5 --save

    # 여러 파일 랜덤 샘플링해서 저장
    python inspect_h5.py path/to/TRAIN_H5/ --sample 9 --save

예시:
    python inspect_h5.py data/TRAIN_H5/LC08_L1GT_160109_20201126_20210316_02_T2_PATCH0.h5
    python inspect_h5.py data/TRAIN_H5/ --sample 6 --save --out output_vis/
"""

import argparse
import glob
import os
import random
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────
# 상수 정의 (patch_dataset.py / qa_pixel_mapping.py 기준)
# ──────────────────────────────────────────────
N_SPECTRAL_BANDS = 8  # B1–B7 + B9 (Cirrus)

BAND_NAMES = ['B1 (Coastal)', 'B2 (Blue)', 'B3 (Green)',
              'B4 (Red)', 'B5 (NIR)', 'B6 (SWIR1)',
              'B7 (SWIR2)', 'B9 (Cirrus)']

CLASS_NAMES = {
    0: 'No-Data',
    1: 'Clear-Sky',
    2: 'Cloud',
    3: 'Shadow',
    4: 'Snow/Ice',
    5: 'Water',
}

# 클래스별 색상 (RGB 0~1)
CLASS_COLORS = {
    0: (0.0,  0.0,  0.0),   # Black  – No-Data
    1: (0.13, 0.55, 0.13),  # Green  – Clear
    2: (1.0,  1.0,  1.0),   # White  – Cloud
    3: (0.5,  0.5,  0.5),   # Grey   – Shadow
    4: (0.0,  1.0,  1.0),   # Cyan   – Snow/Ice
    5: (0.0,  0.0,  1.0),   # Blue   – Water
}

NUM_CLASSES = 6

# Colormap for label maps
CMAP_LABEL = mcolors.ListedColormap([CLASS_COLORS[i] for i in range(NUM_CLASSES)])
NORM_LABEL  = mcolors.BoundaryNorm(boundaries=range(NUM_CLASSES + 1), ncolors=NUM_CLASSES)


# ──────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────

def load_h5(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as hf:
        keys = list(hf.keys())
        assert 'data' in keys, f"'data' key not found. Keys: {keys}"
        data = hf['data'][:]
    return data


def print_h5_info(path: str, data: np.ndarray):
    """텍스트로 H5 내용 요약 출력."""
    print("=" * 60)
    print(f"File : {os.path.basename(path)}")
    print(f"Shape: {data.shape}  (H x W x C)")
    print(f"dtype: {data.dtype}")
    print(f"Size : {data.nbytes / 1024:.1f} KB")
    print()

    n_ch = data.shape[-1]
    print(f"{'Channel':<6} {'Name':<20} {'min':>8} {'max':>8} {'mean':>8} {'nonzero':>10}")
    print("-" * 62)

    for i in range(n_ch):
        ch = data[:, :, i].astype(np.float32)
        if i < N_SPECTRAL_BANDS:
            name = BAND_NAMES[i]
        elif i == N_SPECTRAL_BANDS:
            name = 'QA_PIXEL label'
        elif i == N_SPECTRAL_BANDS + 1:
            name = 'Prediction (opt)'
        elif i == N_SPECTRAL_BANDS + 2:
            name = 'Pseudo-label'
        else:
            name = f'Unknown ch{i}'

        nonzero = int(np.count_nonzero(ch))
        total   = ch.size
        print(f"  ch{i:<3} {name:<20} {ch.min():>8.1f} {ch.max():>8.1f} "
              f"{ch.mean():>8.2f} {nonzero:>6}/{total}")

    # 라벨 클래스 분포
    if n_ch > N_SPECTRAL_BANDS:
        print()
        for label_idx, label_name in [(N_SPECTRAL_BANDS, 'QA_PIXEL'),
                                       (N_SPECTRAL_BANDS + 2, 'Pseudo-label')]:
            if label_idx < n_ch:
                lbl = data[:, :, label_idx].astype(np.uint8)
                total = lbl.size
                print(f"  [{label_name}] 클래스 분포:")
                for cls in range(NUM_CLASSES):
                    cnt = int((lbl == cls).sum())
                    pct = cnt / total * 100
                    bar = '█' * int(pct / 2)
                    print(f"    {cls} {CLASS_NAMES[cls]:<12}: {cnt:>6} px ({pct:5.1f}%) {bar}")
    print("=" * 60)


def normalize_band(arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """퍼센타일 기반 정규화 (0~1)."""
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def make_rgb(data: np.ndarray) -> np.ndarray:
    """B4(R), B3(G), B2(B) → True Color RGB."""
    r = normalize_band(data[:, :, 3])  # B4
    g = normalize_band(data[:, :, 2])  # B3
    b = normalize_band(data[:, :, 1])  # B2
    return np.stack([r, g, b], axis=-1)


def make_false_color(data: np.ndarray) -> np.ndarray:
    """B5(NIR), B4(R), B3(G) → False Color (vegetation)."""
    r = normalize_band(data[:, :, 4])  # B5 NIR
    g = normalize_band(data[:, :, 3])  # B4 Red
    b = normalize_band(data[:, :, 2])  # B3 Green
    return np.stack([r, g, b], axis=-1)


def make_swir_composite(data: np.ndarray) -> np.ndarray:
    """B6(SWIR1), B5(NIR), B4(R) → Snow 탐지용."""
    r = normalize_band(data[:, :, 5])  # B6 SWIR1
    g = normalize_band(data[:, :, 4])  # B5 NIR
    b = normalize_band(data[:, :, 3])  # B4 Red
    return np.stack([r, g, b], axis=-1)


def compute_ndsi(data: np.ndarray) -> np.ndarray:
    """NDSI = (B3 - B6) / (B3 + B6)."""
    g  = data[:, :, 2].astype(np.float32)
    s1 = data[:, :, 5].astype(np.float32)
    denom = g + s1
    denom[denom == 0] = 1e-6
    return (g - s1) / denom


def label_to_rgb(lbl: np.ndarray) -> np.ndarray:
    """라벨 맵 → RGB 이미지."""
    h, w = lbl.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        mask = (lbl == cls)
        rgb[mask] = color
    return rgb


def make_legend_patches():
    return [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
            for c in range(NUM_CLASSES)]


# ──────────────────────────────────────────────
# 메인 시각화
# ──────────────────────────────────────────────

def visualize_patch(path: str, save: bool = False, out_dir: str = '.'):
    data = load_h5(path)

    # 텍스트 출력
    print_h5_info(path, data)

    n_ch = data.shape[-1]
    has_pseudo = (n_ch > N_SPECTRAL_BANDS + 2)

    # ── 그림 레이아웃 구성 ──────────────────────────
    n_cols = 4
    n_rows = 3 if has_pseudo else 2

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(os.path.basename(path), color='white', fontsize=11, y=1.01)

    axes = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            row.append(ax)
        axes.append(row)

    def show(ax, img, title, cmap=None, vmin=None, vmax=None, colorbar=False):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, color='#e0e0ff', fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')
        return im

    # Row 0: 컬러 합성
    show(axes[0][0], make_rgb(data),          'True Color (B4-B3-B2)')
    show(axes[0][1], make_false_color(data),  'False Color (B5-B4-B3)')
    show(axes[0][2], make_swir_composite(data),'SWIR Composite (B6-B5-B4)')

    ndsi = compute_ndsi(data)
    show(axes[0][3], ndsi, 'NDSI\n(Snow/Ice > 0.4)', cmap='RdBu_r', vmin=-1, vmax=1, colorbar=True)

    # Row 1: 스펙트럼 밴드 & 라벨
    band_pairs = [(0, 'B1 Coastal'), (4, 'B5 NIR'), (6, 'B7 SWIR2'), (7, 'B9 Cirrus')]
    for col, (bi, bname) in enumerate(band_pairs):
        arr = normalize_band(data[:, :, bi])
        show(axes[1][col], arr, bname, cmap='gray')

    # Row 2 (항상): QA_PIXEL 라벨
    if n_ch > N_SPECTRAL_BANDS:
        qa_lbl = data[:, :, N_SPECTRAL_BANDS].astype(np.uint8)
        qa_rgb = label_to_rgb(qa_lbl)
        show(axes[-1][0], qa_rgb, 'QA_PIXEL Label')
        axes[-1][0].legend(handles=make_legend_patches(),
                           loc='lower right', fontsize=6,
                           facecolor='#0f0f23', labelcolor='white',
                           framealpha=0.7, handlelength=1.0)

    # Pseudo-label (있을 때)
    if has_pseudo:
        ps_lbl = data[:, :, N_SPECTRAL_BANDS + 2].astype(np.uint8)
        ps_rgb = label_to_rgb(ps_lbl)
        show(axes[-1][1], ps_rgb, 'Pseudo-label')

    # 클래스 픽셀 수 막대그래프
    ax_bar = axes[-1][-2]
    ax_bar.set_facecolor('#16213e')
    if n_ch > N_SPECTRAL_BANDS:
        lbl = data[:, :, N_SPECTRAL_BANDS].astype(np.uint8)
        counts = [(lbl == c).sum() for c in range(NUM_CLASSES)]
        colors_bar = [CLASS_COLORS[c] for c in range(NUM_CLASSES)]
        bars = ax_bar.bar(range(NUM_CLASSES), counts, color=colors_bar, edgecolor='#333355')
        ax_bar.set_xticks(range(NUM_CLASSES))
        ax_bar.set_xticklabels([CLASS_NAMES[c][:5] for c in range(NUM_CLASSES)],
                                rotation=30, fontsize=7, color='#aaaaaa')
        ax_bar.set_title('Class Pixel Count\n(QA_PIXEL)', color='#e0e0ff', fontsize=9, pad=4)
        ax_bar.tick_params(colors='#aaaaaa', labelsize=7)
        ax_bar.set_facecolor('#16213e')
        for spine in ax_bar.spines.values():
            spine.set_edgecolor('#333355')
        ax_bar.yaxis.label.set_color('#aaaaaa')

    # 밴드별 평균값 라인 차트
    ax_line = axes[-1][-1]
    spectral = data[:, :, :N_SPECTRAL_BANDS].astype(np.float32)
    band_means = spectral.mean(axis=(0, 1))
    ax_line.plot(range(N_SPECTRAL_BANDS), band_means,
                 color='#7ec8e3', marker='o', markersize=4, linewidth=1.5)
    ax_line.set_xticks(range(N_SPECTRAL_BANDS))
    ax_line.set_xticklabels(['B1','B2','B3','B4','B5','B6','B7','B9'],
                             fontsize=7, color='#aaaaaa')
    ax_line.set_title('Mean Spectral Values\n(raw DN)', color='#e0e0ff', fontsize=9, pad=4)
    ax_line.set_facecolor('#16213e')
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
        print(f"[saved] {out_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_multiple(h5_dir: str, n_samples: int, save: bool, out_dir: str):
    """디렉토리에서 랜덤 샘플링해서 그리드 시각화."""
    files = glob.glob(os.path.join(h5_dir, '*.h5'))
    if not files:
        print(f"H5 파일을 찾을 수 없습니다: {h5_dir}")
        sys.exit(1)

    sampled = random.sample(files, min(n_samples, len(files)))
    print(f"\n{len(sampled)}개 파일 시각화 중...")

    for f in sampled:
        visualize_patch(f, save=save, out_dir=out_dir)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Landsat H5 패치 내용 확인 및 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('path',
                        help='H5 파일 경로 또는 H5 디렉토리 경로')
    parser.add_argument('--save', action='store_true',
                        help='화면 출력 대신 PNG 파일로 저장')
    parser.add_argument('--out', default='vis_output',
                        help='저장 디렉토리 (기본: vis_output)')
    parser.add_argument('--sample', type=int, default=1,
                        help='디렉토리 지정 시 랜덤 샘플 수 (기본: 1)')
    args = parser.parse_args()

    path = os.path.abspath(args.path)

    if os.path.isdir(path):
        visualize_multiple(path, args.sample, args.save, args.out)
    elif os.path.isfile(path):
        visualize_patch(path, save=args.save, out_dir=args.out)
    else:
        print(f"경로가 없습니다: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
