"""
Full-scene inference utilities for validation confusion matrix evaluation.

Loads bands.tif / cfmask.tif from prepared/ directory and GT labels from
labels/, then runs tiled UNet inference (256×256 tiles, 1-pixel real border)
to produce a full-scene prediction map.
"""

import math
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2


# ── Derived feature helpers (mirrors utils/split_scene.py) ───────────────────
# Copied here to avoid importing split_scene.py's zarr v3 module-level imports.

def _percentile_normalize(arr: np.ndarray, p_lo: int = 2, p_hi: int = 98) -> np.ndarray:
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def compute_rgb(spectral: np.ndarray) -> np.ndarray:
    """(H, W, 8) spectral → percentile-normalised RGB float32 [0,1]. R=ch3(B4), G=ch2(B3), B=ch1(B2)."""
    r = _percentile_normalize(spectral[:, :, 3])
    g = _percentile_normalize(spectral[:, :, 2])
    b = _percentile_normalize(spectral[:, :, 1])
    return np.stack([r, g, b], axis=-1)


def compute_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB float32 [0,1] → HSV float32 [0,1]."""
    rgb_u8 = (rgb * 255).astype(np.uint8)
    hsv_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hsv = hsv_u8.astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv


def compute_sobel(rgb: np.ndarray) -> np.ndarray:
    """RGB float32 [0,1] → Sobel X, Y, Magnitude on luminance. Returns (H, W, 3)."""
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(np.float32)
    sx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return np.stack([sx, sy, mag], axis=-1)

PATCH_SIZE = 256
NODATA = 255


# ── Model loading ─────────────────────────────────────────────────────────────

FILTER_OPTIONS = [16, 32, 24, 32]   # start_filts per stage (mirrors network/model.py)
DEPTH_OPTIONS  = [5,  5,  6,  6]   # UNet depth per stage


def load_model(exp_base: str, stage: int, num_classes: int,
               inp_mode: str, device: torch.device) -> torch.nn.Module:
    """Load a trained UNet from checkpoint (strips DataParallel 'module.' prefix)."""
    from network.unet import UNet
    from dataset.network_input import get_inp_channels
    from utils.dir_paths import EXP_DATA_PATH

    exp_name  = f'{exp_base}_stage{stage}'
    ckpt_path = os.path.join(EXP_DATA_PATH, exp_name, 'model', 'model_best.pth')

    n_inp = get_inp_channels(inp_mode)
    net = UNet(
        num_classes=num_classes,
        in_channels=n_inp,
        depth=DEPTH_OPTIONS[stage],
        start_filts=FILTER_OPTIONS[stage],
        dropout=False,
    )
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', '', 1): v
             for k, v in ckpt['model_state_dict'].items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


# ── Scene data loading ────────────────────────────────────────────────────────

def load_scene_bands(prepared_dir: str) -> np.ndarray:
    """
    Load bands.tif → (H, W, 8) float32 in zarr spectral channel layout:
      ch0=B1(zeros), ch1=B2, ch2=B3, ch3=B4, ch4=B5, ch5=B6, ch6=B7, ch7=B9.

    bands.tif channel order: B2(0), B3(1), B4(2), B5(3), B6(4), B7(5), B9(6), B10(7).
    B1 is zero-filled (absent from bands.tif); B10 (thermal) is unused by any network.
    """
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with rasterio.open(os.path.join(prepared_dir, 'bands.tif')) as src:
            bands = src.read().astype(np.float32)  # (8, H, W) TOA reflectance [0, 1]

    H, W = bands.shape[1], bands.shape[2]
    spectral = np.zeros((H, W, 8), dtype=np.float32)
    spectral[:, :, 1] = bands[0]  # B2
    spectral[:, :, 2] = bands[1]  # B3
    spectral[:, :, 3] = bands[2]  # B4
    spectral[:, :, 4] = bands[3]  # B5
    spectral[:, :, 5] = bands[4]  # B6
    spectral[:, :, 6] = bands[5]  # B7
    spectral[:, :, 7] = bands[6]  # B9
    return spectral  # ch0 (B1) stays zero


def load_cfmask(prepared_dir: str) -> np.ndarray:
    """
    Load cfmask.tif and remap to evaluation space:
      0=no-cloud (clear/snow/water), 1=cloud, 2=shadow, 255=nodata.
    """
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with rasterio.open(os.path.join(prepared_dir, 'cfmask.tif')) as src:
            raw = src.read(1)  # (H, W) uint8

    out = np.zeros_like(raw, dtype=np.uint8)   # 0=no-cloud (clear/snow/water)
    out[raw == 1]   = 1    # cloud
    out[raw == 2]   = 2    # shadow
    out[raw == 255] = 255  # fill / nodata
    return out


def load_gt_labels(labels_dir: str, scene_id: str) -> np.ndarray:
    """
    Load GT napari label tif and remap:
      {4}→1(cloud), {3}→2(shadow), {1,2}→0(no-cloud), {0,255}→255(nodata).
    """
    import rasterio
    label_path = os.path.join(labels_dir, f'{scene_id}_labels.tif')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with rasterio.open(label_path) as src:
            raw = src.read(1)  # (H, W) uint8

    out = np.full_like(raw, NODATA, dtype=np.uint8)
    out[raw == 1] = 0   # water  → no-cloud
    out[raw == 2] = 0   # snow   → no-cloud
    out[raw == 3] = 2   # shadow
    out[raw == 4] = 1   # cloud
    # raw==0 (unlabeled) and raw==255 (fill) → NODATA (already set)
    return out


# ── Inference ─────────────────────────────────────────────────────────────────

def run_scene_inference(net: torch.nn.Module,
                        inp_func,
                        spectral_hw8: np.ndarray,
                        device: torch.device) -> np.ndarray:
    """
    Full-scene tiled inference.

    Tiles: math.ceil(H/256) × math.ceil(W/256).
    Each tile: 258×258 input (256×256 center + 1-pixel real border).
    Derived features (RGB, HSV, Sobel) computed per-tile to match training
    (per-patch percentile normalisation used in split_scene.py).

    Returns (H, W) uint8 prediction array.
    """
    H, W = spectral_hw8.shape[:2]
    result = np.zeros((H, W), dtype=np.uint8)

    n_ty = math.ceil(H / PATCH_SIZE)
    n_tx = math.ceil(W / PATCH_SIZE)

    for iy in range(n_ty):
        for ix in range(n_tx):
            row_s = min(iy * PATCH_SIZE, H - PATCH_SIZE)
            col_s = min(ix * PATCH_SIZE, W - PATCH_SIZE)

            # Compute padded read window (1-pixel real border)
            r0 = max(0, row_s - 1)
            c0 = max(0, col_s - 1)
            r1 = min(H, row_s + PATCH_SIZE + 1)
            c1 = min(W, col_s + PATCH_SIZE + 1)

            spectral_tile = np.zeros((PATCH_SIZE + 2, PATCH_SIZE + 2, 8),
                                     dtype=np.float32)
            off_r = 0 if r0 < row_s else 1  # 0 = border pixel available at top
            off_c = 0 if c0 < col_s else 1
            spectral_tile[off_r:off_r + (r1 - r0),
                          off_c:off_c + (c1 - c0)] = spectral_hw8[r0:r1, c0:c1]

            # Compute derived features per-tile (matches training data generation)
            rgb   = compute_rgb(spectral_tile)    # (258, 258, 3)
            hsv   = compute_hsv(rgb)              # (258, 258, 3)
            sobel = compute_sobel(rgb)            # (258, 258, 3)
            tile_17 = np.concatenate([spectral_tile, rgb, hsv, sobel],
                                     axis=-1)     # (258, 258, 17)

            inp = (torch.from_numpy(tile_17.transpose(2, 0, 1))
                   .unsqueeze(0))                 # (1, 17, 258, 258)
            with torch.no_grad():
                out  = net(inp_func(inp.to(device)))     # (1, n_cls, 258, 258)
                pred = torch.argmax(out, dim=1).squeeze(0)  # (258, 258)

            result[row_s:row_s + PATCH_SIZE,
                   col_s:col_s + PATCH_SIZE] = pred[1:257, 1:257].cpu().numpy()

    return result


# ── Confusion matrix ──────────────────────────────────────────────────────────

def accumulate_cm(cm: np.ndarray, true_arr: np.ndarray,
                  pred_arr: np.ndarray, n_classes: int) -> None:
    """
    Accumulate pixelwise confusion matrix CM[true][pred] in-place.

    Valid pixels: both arrays have value in [0, n_classes-1].
    Excludes nodata (255) and, for n_classes==2, shadow (2) automatically.
    """
    valid = (true_arr < n_classes) & (pred_arr < n_classes)
    t = true_arr[valid].astype(np.int64)
    p = pred_arr[valid].astype(np.int64)
    cm += (np.bincount(n_classes * t + p, minlength=n_classes * n_classes)
           .reshape(n_classes, n_classes))
