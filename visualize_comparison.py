"""
Fmask / 모델 예측 / Ground Truth 3-panel 비교 시각화

사용법:
    python visualize_comparison.py \
        --patch data/VALIDATION_ZARR/LC08_L1GT_188114_20201114_20210315_02_T2_PATCH5.zarr \
        --exp  swirndsi_trial2_stage0 \
        --label_dir label_code/labels \
        [--gpu 0] \
        [--out vis_output/]

    # 여러 패치 한번에 (랜덤 샘플링)
    python visualize_comparison.py \
        --patch data/VALIDATION_ZARR/ \
        --exp  swirndsi_trial2_stage0 \
        --label_dir label_code/labels \
        --sample 6 \
        [--gpu 0] \
        [--out vis_output/]

출력:
    Left   : RGB + Fmask 오버레이 (원본 씬 QA_PIXEL 기반)
    Center : RGB + 모델 예측 오버레이
    Right  : RGB + Ground Truth 오버레이 (수동 라벨)
"""

import argparse
import glob
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
import rasterio
from rasterio import windows as rwin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.network_input import get_inp_func
from network.model import Model
from utils.experiment import Experiment
from utils.qa_pixel_mapping import qa_pixel_to_binary
from utils.split_scene import find_band_file, find_qa_pixel_file
from utils.dir_paths import WEDDELL_SEA_SOURCE_PATH

# ── 상수 ───────────────────────────────────────────────────────────────
PATCH_SIZE    = 256
STRIDE        = 256
MAX_FILL_FRAC = 0.50
MIN_VALID_FRAC = 0.30
VALID_LABEL_VALUES = {1, 2, 3, 4}   # scene_to_patches.py 와 동일

COLORS = {          # (R, G, B) 0-1 범위
    'cloud':    (1.00, 1.00, 1.00),   # 흰색
    'no_cloud': (0.13, 0.55, 0.13),   # 초록
    'ignore':   (0.15, 0.15, 0.15),   # 진회색
}
ALPHA = 0.55   # 오버레이 불투명도


# ── Gini 지수 계산 ────────────────────────────────────────────────────

def compute_gini(patch_path: str) -> float:
    """
    GT label의 3-class Gini impurity를 반환.
    클래스: 0(no-cloud), 1(cloud), 255(ignore)
    gini = 1 - Σp_i²,  범위 [0, 2/3]  (2/3 ≈ 0.667 = 완전한 1/3:1/3:1/3 혼합)
    값이 높을수록 세 레이블이 고르게 섞인 패치.
    """
    store = zarr.open_group(patch_path, mode='r')
    label = store['label'][:]
    n = label.size
    if n == 0:
        return 0.0
    gini = 1.0
    for v in (0, 1, 255):
        p = float((label == v).sum()) / n
        gini -= p * p
    return gini


def sample_by_gini(patch_dirs: list[str], n: int,
                   min_gini: float) -> list[str]:
    """
    min_gini 이상인 패치를 필터링한 뒤 최대 n개 랜덤 샘플링.
    필터 통과 패치가 n보다 적으면 전부 반환.
    """
    qualified = [p for p in patch_dirs if compute_gini(p) >= min_gini]
    print(f"  Gini ≥ {min_gini:.2f}: {len(qualified)}/{len(patch_dirs)} 패치 통과")
    return random.sample(qualified, min(n, len(qualified)))


# ── 패치 이름 파싱 ─────────────────────────────────────────────────────

def parse_patch_name(patch_path: str):
    """'..._PATCH42.zarr'  →  (scene_id, 42)"""
    stem = Path(patch_path).stem
    m = re.match(r'^(.+)_PATCH(\d+)$', stem)
    if not m:
        raise ValueError(f"패치 이름 파싱 실패: {stem}")
    return m.group(1), int(m.group(2))


# ── 씬 디렉토리 탐색 ───────────────────────────────────────────────────

def find_scene_dir(scene_id: str) -> Path:
    """scene_id 에서 날짜를 파싱해 Weddell Sea 경로를 반환."""
    # LC08_L1GT_188114_20201114_20210315_02_T2  →  date = 20201114
    parts = scene_id.split('_')
    date_str = parts[3]
    year     = date_str[:4]
    month    = date_str[4:6]
    candidate = Path(WEDDELL_SEA_SOURCE_PATH) / year / month / date_str / scene_id
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"씬 디렉토리를 찾을 수 없습니다: {candidate}\n"
        f"--scene_dir 로 직접 지정하거나 WEDDELL_SEA_SOURCE_PATH 를 확인하세요."
    )


# ── 패치 좌표 재계산 (scene 재스캔) ────────────────────────────────────

def find_patch_coords(scene_dir: Path, label_path: Path,
                      target_idx: int) -> tuple[int, int]:
    """
    scene_to_patches.py 와 동일한 필터링 로직으로 패치 좌표를 찾는다.
    target_idx 번째로 저장됐을 패치의 (row_start, col_start) 를 반환.
    """
    qa_file = find_qa_pixel_file(str(scene_dir))
    if qa_file is None:
        raise FileNotFoundError(f"QA_PIXEL 파일 없음: {scene_dir}")

    with rasterio.open(label_path) as src:
        labels_raw = src.read(1).astype(np.uint8)

    with rasterio.open(qa_file) as src:
        H, W = src.height, src.width

    n_saved = 0
    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):
            lr = labels_raw[i:i + PATCH_SIZE, j:j + PATCH_SIZE]

            with rasterio.open(qa_file) as src:
                win = rwin.Window(j, i, PATCH_SIZE, PATCH_SIZE)
                qa_patch = src.read(1, window=win).astype(np.uint16)

            fill_frac  = float((qa_patch & 1).astype(bool).mean())
            if fill_frac > MAX_FILL_FRAC:
                continue
            valid_frac = float(np.isin(lr, list(VALID_LABEL_VALUES)).mean())
            if valid_frac < MIN_VALID_FRAC:
                continue

            if n_saved == target_idx:
                return i, j
            n_saved += 1

    raise ValueError(
        f"patch_idx={target_idx} 를 씬에서 찾을 수 없습니다. "
        f"scene_to_patches.py 와 파라미터가 다를 수 있습니다."
    )


# ── QA_PIXEL → Fmask binary 로드 ──────────────────────────────────────

def load_fmask(scene_dir: Path, row: int, col: int) -> np.ndarray:
    """원본 씬의 QA_PIXEL에서 (row, col) 위치의 패치를 이진 마스크로 반환."""
    qa_file = find_qa_pixel_file(str(scene_dir))
    with rasterio.open(qa_file) as src:
        win = rwin.Window(col, row, PATCH_SIZE, PATCH_SIZE)
        qa  = src.read(1, window=win).astype(np.uint16)
    return qa_pixel_to_binary(qa)   # 0=no-cloud, 1=cloud, 255=no-data


# ── 모델 추론 ──────────────────────────────────────────────────────────

def run_inference(patch_path: str, exp_name: str, gpu_id: list) -> np.ndarray:
    """패치 한 개에 대해 모델 예측을 실행. (256, 256) uint8 반환."""
    import argparse as _ap
    args = _ap.Namespace(
        exp_name=exp_name, stage=3, full=False, dropout=True,
        learning_rate=1e-6, inp_mode='swirndsi', bands=None, indices=None,
    )
    exp   = Experiment(args, mode='test')
    model = Model(exp, gpu_id=gpu_id)
    model.network.eval()

    store    = zarr.open_group(patch_path, mode='r')
    spectral = store['spectral'][:].astype(np.float32) / 10000.0
    rgb_arr  = store['rgb'][:]
    hsv      = store['hsv'][:]
    sobel    = store['sobel'][:]

    full = np.concatenate([spectral, rgb_arr, hsv, sobel], axis=-1)
    full = np.pad(full, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
    full = np.transpose(full, (2, 0, 1))
    inp  = torch.from_numpy(full[None]).float()

    inp_func = get_inp_func(exp.inp_mode)
    device   = next(model.network.parameters()).device
    with torch.no_grad():
        out  = model.network(inp_func(inp.to(device)))
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)[0].cpu().numpy()

    return pred[1:-1, 1:-1].astype(np.uint8)   # 패딩 제거


# ── 씬 오버뷰 + 패치 위치 표시 ────────────────────────────────────────

def make_scene_overview(scene_dir: Path, row: int, col: int,
                         patch_size: int = PATCH_SIZE) -> np.ndarray | None:
    """
    씬의 FCI (B5-NIR / B4-Red / B3-Green) 썸네일에 패치 위치를 빨간 박스로 표시.
    반환: (H', W', 3) float32 ∈ [0,1]  또는  None (밴드 파일 없을 시)
    """
    # FCI 밴드: R=B5(NIR), G=B4(Red), B=B3(Green)
    fci_bands = ['B5', 'B4', 'B3']
    band_files = {bk: find_band_file(str(scene_dir), bk) for bk in fci_bands}
    if any(v is None for v in band_files.values()):
        return None

    with rasterio.open(band_files['B5']) as src:
        H, W = src.height, src.width
    max_px = 800
    scale  = max(1, max(H, W) // max_px)
    out_h, out_w = H // scale, W // scale

    channels = []
    for bk in fci_bands:
        with rasterio.open(band_files[bk]) as src:
            ch = src.read(1, out_shape=(out_h, out_w)).astype(np.float32)
        valid = ch[ch > 0]
        if valid.size:
            p2, p98 = np.percentile(valid, [2, 98])
            ch = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
        else:
            ch = np.zeros_like(ch)
        channels.append(ch)

    thumb = np.stack(channels, axis=-1)  # (H', W', 3) FCI

    # 패치 좌표를 썸네일 스케일로 변환
    r0 = min(row // scale,        out_h - 1)
    c0 = min(col // scale,        out_w - 1)
    r1 = min((row + patch_size) // scale, out_h - 1)
    c1 = min((col + patch_size) // scale, out_w - 1)
    t  = max(2, scale // 2)   # 선 두께

    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    thumb[r0:r1, c0:min(c0+t, out_w)] = red   # 왼쪽
    thumb[r0:r1, max(c1-t, 0):c1]     = red   # 오른쪽
    thumb[r0:min(r0+t, out_h), c0:c1] = red   # 위
    thumb[max(r1-t, 0):r1, c0:c1]     = red   # 아래

    return np.clip(thumb, 0, 1)


# ── 오버레이 생성 ──────────────────────────────────────────────────────

def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    RGB 이미지 위에 이진 마스크를 반투명 오버레이로 합성.
    mask: 0=no-cloud, 1=cloud, 255=ignore
    """
    out = rgb.copy()
    for val, key in [(1, 'cloud'), (0, 'no_cloud'), (255, 'ignore')]:
        m = (mask == val)
        if m.any():
            color = np.array(COLORS[key], dtype=np.float32)
            out[m] = out[m] * (1 - ALPHA) + color * ALPHA
    return np.clip(out, 0, 1)


# ── 단일 패치 시각화 ───────────────────────────────────────────────────

def visualize_patch(patch_path: str, exp_name: str, label_dir: str,
                    gpu_id: list, out_dir: str,
                    scene_dir_override: str = None):

    scene_id, patch_idx = parse_patch_name(patch_path)
    print(f"[{scene_id}  PATCH{patch_idx}]")

    # ── 씬 디렉토리 ──
    scene_dir = Path(scene_dir_override) if scene_dir_override \
                else find_scene_dir(scene_id)
    print(f"  씬 디렉토리: {scene_dir}")

    # ── 라벨 파일 ──
    label_path = Path(label_dir) / f"{scene_id}_labels.tif"
    if not label_path.exists():
        raise FileNotFoundError(f"라벨 파일 없음: {label_path}")

    # ── 패치 좌표 (재스캔) ──
    print(f"  패치 좌표 탐색 중 (PATCH{patch_idx})...")
    row, col = find_patch_coords(scene_dir, label_path, patch_idx)
    print(f"  → row={row}, col={col}")

    # ── Fmask, 모델 예측, GT 로드 ──
    fmask = load_fmask(scene_dir, row, col)
    pred  = run_inference(patch_path, exp_name, gpu_id)

    store = zarr.open_group(patch_path, mode='r')
    rgb   = store['rgb'][:]                    # (H, W, 3) float32 ∈ [0,1]
    gt    = store['label'][:]                  # (H, W) uint8

    # swath 밖 fill 픽셀은 모델이 0(no-cloud)으로 예측하는 경향이 있으므로
    # QA_PIXEL fill 마스크만 사용해 prediction에도 255(ignore)로 덮어씌움
    # (gt==255는 미라벨 영역도 포함하므로 사용하지 않음)
    pred[fmask == 255] = 255

    # ── 플롯 (2×3 그리드) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=120)

    legend_patches = [
        mpatches.Patch(color=COLORS['cloud'],    label='Cloud'),
        mpatches.Patch(color=COLORS['no_cloud'], label='No-Cloud'),
        mpatches.Patch(color=COLORS['ignore'],   label='Ignore/No-Data'),
    ]

    # 상단 행: Fmask / 모델 예측 / GT
    for ax, title, mask in zip(
            axes[0],
            ['Fmask (QA_PIXEL)', f'Model Prediction\n({exp_name})', 'Ground Truth (수동 라벨)'],
            [fmask, pred, gt]):
        ax.imshow(overlay_mask(rgb, mask))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
                  framealpha=0.8)

    # 하단 행 (1,0) / (1,2): 비움
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')

    # 하단 행 (1,1): 씬 오버뷰 + 패치 위치
    overview = make_scene_overview(scene_dir, row, col)
    if overview is not None:
        axes[1, 1].imshow(overview)
        axes[1, 1].set_title(
            f'Scene FCI Overview  (row={row}, col={col})', fontsize=11)
    axes[1, 1].axis('off')

    fig.suptitle(f"{scene_id}  PATCH{patch_idx}  (row={row}, col={col})",
                 fontsize=10)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_id}_PATCH{patch_idx}_{exp_name}_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Fmask / 모델 / GT 3-panel 비교')
    p.add_argument('--patch',     required=True,
                   help='zarr 패치 경로 또는 VALIDATION_ZARR 디렉토리')
    p.add_argument('--exp',       required=True,
                   help='실험 이름 (e.g. swirndsi_trial2_stage0)')
    p.add_argument('--label_dir', default='label_code/labels',
                   help='수동 라벨 TIF 디렉토리 (기본: label_code/labels)')
    p.add_argument('--scene_dir', default=None,
                   help='씬 디렉토리 직접 지정 (생략 시 자동 탐색)')
    p.add_argument('--gpu',       type=int, nargs='+', default=[0])
    p.add_argument('--out',       default='vis_output/',
                   help='출력 디렉토리 (기본: vis_output/)')
    p.add_argument('--sample',    type=int, default=None,
                   help='디렉토리 지정 시 랜덤 샘플 수')
    p.add_argument('--min_gini', type=float, default=0.0,
                   help='최소 Gini impurity 기준 (기본 0.0=필터 없음). '
                        '3-class Gini: {0,1,255} 기준, 최대 0.667. '
                        '권장: 0.1~0.3')
    p.add_argument('--list_only', action='store_true', default=False,
                   help='패치 경로만 출력하고 추론은 실행하지 않음 (bash 파이프용)')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 단일 패치 or 디렉토리
    if args.patch.endswith('.zarr'):
        targets = [args.patch]
    else:
        all_patches = sorted(glob.glob(os.path.join(args.patch, '*.zarr')))
        if args.sample and args.min_gini > 0.0:
            targets = sample_by_gini(all_patches, args.sample, args.min_gini)
        elif args.sample:
            targets = random.sample(all_patches, min(args.sample, len(all_patches)))
        elif args.min_gini > 0.0:
            targets = [p for p in all_patches if compute_gini(p) >= args.min_gini]
        else:
            targets = all_patches

    if args.list_only:
        for t in targets:
            print(t)
        sys.exit(0)

    label_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.label_dir)

    for t in targets:
        try:
            visualize_patch(t, args.exp, label_dir, args.gpu,
                            args.out, args.scene_dir)
        except Exception as e:
            print(f"  [오류] {t}: {e}")
