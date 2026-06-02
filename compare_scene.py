"""
씬 전체 비교 시각화: Fmask | 모델 예측 | Ground Truth

씬 전체를 256×256 패치로 분할해 모델 추론 후, 씬 전체 지도로 합쳐서 시각화.
수동 라벨이 있는 validation 씬(6개)에서 사용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예시

  # GT 있는 val 씬 (fmask.png + model_*.png + ground_truth.png 생성)
    cd /home/pyuncb/src
    conda run -n remote python compare_scene.py \
        --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
        --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
        --exp swirndsi_trial2_stage3

  # GT 없는 train 씬 (fmask.png + model_*.png 만 생성)
    conda run -n remote python compare_scene.py \
        --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188115_20201114_20210315_02_T2 \
        --exp swirndsi_trial2_stage3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
인자 설명:

  --scene_dir   원본 Landsat L1 씬 폴더 (*_B1.TIF 등이 있는 폴더)
                경로 패턴: {WEDDELL}/{year}/{month}/{date}/{scene_id}/
                WEDDELL = /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

  --label_path  수동 라벨 GeoTIFF (label_code/labels/ 아래에 위치)
                파일명 패턴: {scene_id}_labels.tif

  --exp         실험 이름 (exp_data/ 아래 폴더명)
                현재 사용 가능한 실험:
                  swirndsi_trial2_stage0  (stage 0, 가장 기본)
                  swirndsi_trial2_stage1
                  swirndsi_trial2_stage2
                  swirndsi_trial2_stage3  (최신, 권장)

  --gpu         GPU ID (기본: 0). 멀티 GPU: --gpu 0 1
  --out         결과 저장 디렉토리 (기본: vis_output/)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
라벨 가능한 6개 val 씬 목록 (scene_dir / label_path 쌍):

  165110  2020/03/20200302  LC08_L1GT_165110_20200302_20201016_02_T2
  171110  2020/02/20200225  LC08_L1GT_171110_20200225_20201016_02_T2
  177110  2020/02/20200219  LC08_L1GT_177110_20200219_20201016_02_T2
  181098  2020/04/20200419  LC08_L1GT_181098_20200419_20201016_02_T2
  188114  2020/11/20201114  LC08_L1GT_188114_20201114_20210315_02_T2
  199110  2020/01/20200128  LC08_L1GT_199110_20200128_20201016_02_T2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력:
  vis_output/{scene_id}/fmask.png             — Fmask 오버레이
  vis_output/{scene_id}/model_{exp}.png       — 모델 예측 오버레이
  vis_output/{scene_id}/ground_truth.png      — GT 오버레이 (--label_path 지정 시만)
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
    _dn_to_toa_uint16, _load_sun_sin,
)

PATCH_SIZE = 256
STRIDE     = 256
NODATA_VAL = 255

COLORS = {
    0:   (0.13, 0.55, 0.13),   # 초록    — no-cloud
    1:   (0.53, 0.81, 0.98),   # 하늘색  — cloud
    2:   (0.72, 0.53, 0.90),   # 연보라색 — cloud shadow
    255: (0.10, 0.10, 0.10),   # 진회색  — ignore/no-data
}
ALPHA = 0.8


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

def _detect_num_classes(exp_name: str) -> int:
    """체크포인트의 conv_final.weight shape에서 num_classes 자동 감지."""
    import torch as _torch
    from utils.dir_paths import EXP_DATA_PATH as _EDP
    pth = os.path.join(_EDP, exp_name, 'model', 'model_best.pth')
    if not os.path.exists(pth):
        return 3
    ckpt = _torch.load(pth, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', {})
    for key in ('module.conv_final.weight', 'conv_final.weight'):
        if key in state:
            return state[key].shape[0]
    return 3


def run_scene_inference(spectral: np.ndarray, exp_name: str,
                        gpu_id: list,
                        stage: int = 3,
                        inp_mode: str = 'swirndsi') -> np.ndarray:
    """
    씬 전체를 256×256 패치로 분할해 모델 추론 후 전체 예측 맵 반환.
    반환: (H, W) uint8  — 0=no-cloud, 1=cloud, 255=처리 안 된 영역
    """
    import argparse as _ap
    num_classes = _detect_num_classes(exp_name)
    print(f"  num_classes: {num_classes}")
    args = _ap.Namespace(
        exp_name=exp_name, stage=stage, full=False, dropout=True,
        learning_rate=1e-6, inp_mode=inp_mode, bands=None, indices=None,
        num_classes=num_classes,
    )
    exp   = Experiment(args, mode='test')
    model = Model(exp, gpu_id=gpu_id)
    model.network.eval()
    device   = next(model.network.parameters()).device
    inp_func = get_inp_func(exp.inp_mode)

    H, W   = spectral.shape[:2]
    pred_map = np.full((H, W), NODATA_VAL, dtype=np.uint8)

    # 마지막 패치가 오른쪽/아래 가장자리까지 커버하도록 끝값 보장
    def make_coords(length: int) -> list[int]:
        coords = list(range(0, length - PATCH_SIZE + 1, STRIDE))
        if not coords or coords[-1] + PATCH_SIZE < length:
            coords.append(length - PATCH_SIZE)
        return coords

    rows = make_coords(H)
    cols = make_coords(W)
    total = len(rows) * len(cols)

    print(f"  모델 추론 ({total} 패치)...")
    with torch.no_grad():
        for i in tqdm(rows, leave=False):
            for j in cols:
                # 학습 시와 동일하게 258×258 추출 (1px 실제 인접 픽셀)
                ri0 = max(0, i - 1);  ri1 = min(H, i + PATCH_SIZE + 1)
                ci0 = max(0, j - 1);  ci1 = min(W, j + PATCH_SIZE + 1)
                off_r = 1 if ri0 == i else 0  # 1 when top border unavailable
                off_c = 1 if ci0 == j else 0  # 1 when left border unavailable

                patch_pad = np.zeros(
                    (PATCH_SIZE + 2, PATCH_SIZE + 2, spectral.shape[2]),
                    dtype=spectral.dtype)
                patch_pad[off_r:off_r + (ri1 - ri0),
                          off_c:off_c + (ci1 - ci0)] = spectral[ri0:ri1, ci0:ci1]

                rgb_p   = compute_rgb(patch_pad)
                hsv_p   = compute_hsv(rgb_p)
                sobel_p = compute_sobel(rgb_p)

                full = np.concatenate(
                    [patch_pad.astype(np.float32) / 10000.0, rgb_p, hsv_p, sobel_p],
                    axis=-1,
                )  # (258, 258, 17) — zero padding 없음

                inp = torch.from_numpy(
                    np.transpose(full, (2, 0, 1))[None]
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

_LABEL_REMAP = {0: 255, 1: 0, 2: 0, 3: 2, 4: 1, 255: 255}

def remap_gt(labels_raw: np.ndarray) -> np.ndarray:
    out = np.full_like(labels_raw, 255, dtype=np.uint8)
    for src_val, dst_val in _LABEL_REMAP.items():
        out[labels_raw == src_val] = dst_val
    return out


# ── 메인 ─────────────────────────────────────────────────────────────

def _save_panel(rgb_v: np.ndarray, mask: np.ndarray,
                title: str, out_path: str, suptitle: str):
    """단일 패널 이미지를 파일로 저장."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
    legend_patches = [
        mpatches.Patch(color=COLORS[1],   label='Cloud'),
        mpatches.Patch(color=COLORS[2],   label='Cloud Shadow'),
        mpatches.Patch(color=COLORS[0],   label='No-Cloud'),
        mpatches.Patch(color=COLORS[255], label='Ignore/No-Data'),
    ]
    ax.imshow(overlay_mask(rgb_v, mask))
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=9, framealpha=0.8)
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def compare_scene(scene_dir: str, exp_name: str,
                  gpu_id: list, out_dir: str,
                  label_path: str = None,
                  stage: int = 3,
                  inp_mode: str = 'swirndsi'):

    scene_id = Path(scene_dir).name
    print(f"\n[{scene_id}]")

    # ── 데이터 로드 ──
    print("  밴드 로딩...")
    spectral = load_scene_bands(scene_dir)
    H, W     = spectral.shape[:2]
    print(f"  씬 크기: {H} × {W}")

    # DN → TOA reflectance ×10000 (학습 파이프라인과 동일)
    sun_sin = _load_sun_sin(scene_dir)
    print(f"  TOA 변환 (sun_sin={sun_sin:.4f})...")
    spectral = _dn_to_toa_uint16(spectral, sun_sin=sun_sin)

    print("  RGB 생성...")
    rgb = load_scene_rgb(spectral)

    print("  QA_PIXEL (Fmask) 로딩...")
    qa_file = find_qa_pixel_file(scene_dir)
    with rasterio.open(qa_file) as src:
        qa = src.read(1).astype(np.uint16)
    fmask = qa_pixel_to_binary(qa)

    # ── 모델 추론 ──
    pred = run_scene_inference(spectral, exp_name, gpu_id,
                               stage=stage, inp_mode=inp_mode)
    pred[fmask == 255] = 255   # fill 픽셀은 모델 예측 무시, no-data로 표시

    # ── 다운샘플링 ──
    scale = max(1, H // 2000)
    if scale > 1:
        def ds(arr):
            return arr[::scale, ::scale]
        rgb_v, fmask_v, pred_v = ds(rgb), ds(fmask), ds(pred)
        print(f"  표시 해상도: {rgb_v.shape[:2]} (1/{scale} 다운샘플)")
    else:
        rgb_v, fmask_v, pred_v = rgb, fmask, pred

    # ── 씬별 출력 폴더 ──
    scene_out = os.path.join(out_dir, scene_id)
    os.makedirs(scene_out, exist_ok=True)

    # ── Fmask 저장 ──
    fmask_path = os.path.join(scene_out, "fmask.png")
    _save_panel(rgb_v, fmask_v, 'Fmask (QA_PIXEL)', fmask_path, scene_id)
    print(f"  저장: {fmask_path}")

    # ── Model Prediction 저장 ──
    pred_path = os.path.join(scene_out, f"model_{exp_name}.png")
    _save_panel(rgb_v, pred_v, f'Model: {exp_name}', pred_path, scene_id)
    print(f"  저장: {pred_path}")

    # ── GT 저장 (label_path 제공 시에만) ──
    if label_path:
        print("  GT 라벨 로딩...")
        with rasterio.open(label_path) as src:
            labels_raw = src.read(1).astype(np.uint8)
        gt = remap_gt(labels_raw)
        gt_v = gt[::scale, ::scale] if scale > 1 else gt
        gt_path = os.path.join(scene_out, "ground_truth.png")
        _save_panel(rgb_v, gt_v, 'Ground Truth (수동 라벨)', gt_path, scene_id)
        print(f"  저장: {gt_path}")


# ── inp_mode 자동 감지 ────────────────────────────────────────────────

def _detect_inp_mode(exp_name: str) -> str:
    """체크포인트에서 inp_mode를 읽어 반환. 구버전(함수명 형식)도 역매핑 처리."""
    import torch
    from utils.dir_paths import EXP_DATA_PATH
    from dataset.network_input import _PRESET_MODES

    ckpt_path = os.path.join(EXP_DATA_PATH, exp_name, 'model', 'model_best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('inp_mode', 'swirndsi')

    # 구버전 호환: 함수명(inp_xxx)으로 저장된 경우 모드 키로 역매핑
    if saved.startswith('inp_'):
        for key, (fn, _) in _PRESET_MODES.items():
            if fn.__name__ == saved:
                return key
        raise ValueError(f'체크포인트의 inp_mode "{saved}"를 _PRESET_MODES에서 찾을 수 없음')

    return saved


# ── CLI ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='씬 전체 Fmask/모델(/GT) 비교')
    p.add_argument('--scene_dir',   required=True)
    p.add_argument('--label_path',  default=None,
                   help='수동 라벨 GeoTIFF 경로 (생략 시 GT 패널 미생성 — train 씬에 사용)')
    p.add_argument('--exp',         required=True,
                   help='실험 이름 (e.g. swirndsi_trial2_stage0)')
    p.add_argument('--stage',       type=int, default=3,
                   help='모델 스테이지 (0-3, 네트워크 구조 결정)')
    p.add_argument('--inp_mode',    default=None,
                   help='입력 모드 (생략 시 체크포인트에서 자동 감지)')
    p.add_argument('--gpu',         type=int, nargs='+', default=[0])
    p.add_argument('--out',         default='vis_output/')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    inp_mode = args.inp_mode or _detect_inp_mode(args.exp)
    print(f"  inp_mode: {inp_mode}")
    compare_scene(
        scene_dir  = args.scene_dir,
        exp_name   = args.exp,
        gpu_id     = args.gpu,
        out_dir    = args.out,
        label_path = args.label_path,
        stage      = args.stage,
        inp_mode   = inp_mode,
    )
