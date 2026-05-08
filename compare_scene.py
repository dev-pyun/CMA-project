"""
씬 전체 비교 시각화: Fmask | 모델 예측 | Ground Truth

Usage:
    python compare_scene.py \
        --scene_dir /earth00_home/.../LC08_L1GT_188114_20201114_20210315_02_T2 \
        --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
        --exp swirndsi_trial2_stage0 \
        [--gpu 0] \
        [--out vis_output/]

출력:
    Left   : RGB + Fmask 오버레이
    Center : RGB + 모델 예측 오버레이
    Right  : RGB + Ground Truth 오버레이 (수동 라벨)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.network_input import get_inp_func
from network.model import Model
from utils.experiment import Experiment
from utils.qa_pixel_mapping import qa_pixel_to_binary
from utils.split_scene import (
    ALL_BANDS, N_SPECTRAL,
    find_band_file, find_qa_pixel_file,
    compute_rgb, compute_hsv, compute_sobel,
    _percentile_normalize,
)

PATCH_SIZE = 256
STRIDE     = 256
NODATA_VAL = 255

COLORS = {
    0:   (0.13, 0.55, 0.13),   # 초록  — no-cloud
    1:   (1.00, 1.00, 1.00),   # 흰색  — cloud
    255: (0.10, 0.10, 0.10),   # 진회색 — ignore/no-data
}
ALPHA = 0.55


# ── 씬 전체 밴드 로드 ──────────────────────────────────────────────────

def load_scene_bands(scene_dir: str) -> np.ndarray:
    """B1-B7, B9 전체를 (H, W, 8) uint16 로 반환."""
    band_files = {}
    for bk in ALL_BANDS:
        bf = find_band_file(scene_dir, bk)
        if bf:
            band_files[bk] = bf
        elif bk != 'B9':
            raise FileNotFoundError(f"필수 밴드 {bk} 없음: {scene_dir}")

    with rasterio.open(list(band_files.values())[0]) as src:
        H, W = src.height, src.width

    spectral = np.zeros((H, W, N_SPECTRAL), dtype=np.uint16)
    print("  밴드 로딩...")
    for ch, bk in enumerate(tqdm(ALL_BANDS, leave=False)):
        if bk not in band_files:
            continue
        with rasterio.open(band_files[bk]) as src:
            spectral[:, :, ch] = src.read(1)

    return spectral


def load_scene_rgb(spectral: np.ndarray) -> np.ndarray:
    """씬 전체에 대한 percentile 정규화 RGB (H, W, 3) float32."""
    r = _percentile_normalize(spectral[:, :, 3])  # B4
    g = _percentile_normalize(spectral[:, :, 2])  # B3
    b = _percentile_normalize(spectral[:, :, 1])  # B2
    return np.stack([r, g, b], axis=-1)


# ── 모델 추론 (씬 전체 패치 스캔) ─────────────────────────────────────

def run_scene_inference(spectral: np.ndarray, exp_name: str,
                        gpu_id: list) -> np.ndarray:
    """
    씬 전체를 256×256 패치로 분할해 모델 추론 후 전체 예측 맵 반환.
    반환: (H, W) uint8  — 0=no-cloud, 1=cloud, 255=처리 안 된 영역
    """
    import argparse as _ap
    args = _ap.Namespace(
        exp_name=exp_name, stage=3, full=False, dropout=True,
        learning_rate=1e-6, inp_mode='swirndsi', bands=None, indices=None,
    )
    exp   = Experiment(args, mode='test')
    model = Model(exp, gpu_id=gpu_id)
    model.network.eval()
    device   = next(model.network.parameters()).device
    inp_func = get_inp_func(exp.inp_mode)

    H, W   = spectral.shape[:2]
    pred_map = np.full((H, W), NODATA_VAL, dtype=np.uint8)

    rows = list(range(0, H - PATCH_SIZE + 1, STRIDE))
    cols = list(range(0, W - PATCH_SIZE + 1, STRIDE))
    total = len(rows) * len(cols)

    print(f"  모델 추론 ({total} 패치)...")
    with torch.no_grad():
        for i in tqdm(rows, leave=False):
            for j in cols:
                patch = spectral[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

                # 패치별 파생 피처 계산 (학습 시와 동일)
                rgb_p   = compute_rgb(patch)
                hsv_p   = compute_hsv(rgb_p)
                sobel_p = compute_sobel(rgb_p)

                full = np.concatenate(
                    [patch.astype(np.float32) / 10000.0, rgb_p, hsv_p, sobel_p],
                    axis=-1,
                )  # (256, 256, 17)

                # 1px 패딩 → 모델 입력
                full = np.pad(full, ((1,1),(1,1),(0,0)), constant_values=0)
                inp  = torch.from_numpy(
                    np.transpose(full, (2,0,1))[None]
                ).float()

                out  = model.network(inp_func(inp.to(device)))
                p    = torch.argmax(F.softmax(out, dim=1), dim=1)[0]
                pred_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = \
                    p[1:-1, 1:-1].cpu().numpy().astype(np.uint8)

    return pred_map


# ── 마스크 오버레이 합성 ───────────────────────────────────────────────

def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for val, color in COLORS.items():
        m = (mask == val)
        if m.any():
            c = np.array(color, dtype=np.float32)
            out[m] = out[m] * (1 - ALPHA) + c * ALPHA
    return np.clip(out, 0, 1)


# ── GT 라벨 remap (scene_to_patches.py 와 동일) ───────────────────────

_LABEL_REMAP = {0: 255, 1: 0, 2: 0, 3: 1, 4: 1, 255: 255}

def remap_gt(labels_raw: np.ndarray) -> np.ndarray:
    out = np.full_like(labels_raw, 255, dtype=np.uint8)
    for src_val, dst_val in _LABEL_REMAP.items():
        out[labels_raw == src_val] = dst_val
    return out


# ── 메인 ─────────────────────────────────────────────────────────────

def compare_scene(scene_dir: str, label_path: str, exp_name: str,
                  gpu_id: list, out_dir: str):

    scene_id = Path(scene_dir).name
    print(f"\n[{scene_id}]")

    # ── 데이터 로드 ──
    print("  밴드 로딩...")
    spectral = load_scene_bands(scene_dir)
    H, W     = spectral.shape[:2]
    print(f"  씬 크기: {H} × {W}")

    print("  RGB 생성...")
    rgb = load_scene_rgb(spectral)

    print("  QA_PIXEL (Fmask) 로딩...")
    qa_file = find_qa_pixel_file(scene_dir)
    with rasterio.open(qa_file) as src:
        qa = src.read(1).astype(np.uint16)
    fmask = qa_pixel_to_binary(qa)

    print("  GT 라벨 로딩...")
    with rasterio.open(label_path) as src:
        labels_raw = src.read(1).astype(np.uint8)
    gt = remap_gt(labels_raw)

    # ── 모델 추론 ──
    pred = run_scene_inference(spectral, exp_name, gpu_id)

    # ── 시각화 ──
    print("  플롯 생성...")
    # 큰 씬이므로 다운샘플링해서 저장
    scale = max(1, H // 2000)   # 최대 ~2000px
    if scale > 1:
        def ds(arr):
            return arr[::scale, ::scale]
        rgb_v, fmask_v, pred_v, gt_v = ds(rgb), ds(fmask), ds(pred), ds(gt)
        print(f"  표시 해상도: {rgb_v.shape[:2]} (1/{scale} 다운샘플)")
    else:
        rgb_v, fmask_v, pred_v, gt_v = rgb, fmask, pred, gt

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=120)
    legend_patches = [
        mpatches.Patch(color=COLORS[1],   label='Cloud'),
        mpatches.Patch(color=COLORS[0],   label='No-Cloud'),
        mpatches.Patch(color=COLORS[255], label='Ignore/No-Data'),
    ]
    for ax, title, mask in zip(
            axes,
            ['Fmask (QA_PIXEL)', f'Model: {exp_name}', 'Ground Truth'],
            [fmask_v, pred_v, gt_v]):
        ax.imshow(overlay_mask(rgb_v, mask))
        ax.set_title(title, fontsize=13)
        ax.axis('off')
        ax.legend(handles=legend_patches, loc='lower right',
                  fontsize=9, framealpha=0.8)

    fig.suptitle(scene_id, fontsize=11)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_id}_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='씬 전체 Fmask/모델/GT 비교')
    p.add_argument('--scene_dir',   required=True)
    p.add_argument('--label_path',  required=True,
                   help='수동 라벨 GeoTIFF 경로')
    p.add_argument('--exp',         required=True,
                   help='실험 이름 (e.g. swirndsi_trial2_stage0)')
    p.add_argument('--gpu',         type=int, nargs='+', default=[0])
    p.add_argument('--out',         default='vis_output/')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    compare_scene(
        scene_dir  = args.scene_dir,
        label_path = args.label_path,
        exp_name   = args.exp,
        gpu_id     = args.gpu,
        out_dir    = args.out,
    )
