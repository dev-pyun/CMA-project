"""
임계값 기반 cloud segmentation 테스트.

prepared 씬을 로드해 3가지 레이어를 napari에서 비교:
  - CFMask ref    : QA_PIXEL 기반 (기준)
  - Threshold Seg : 밝기/NDSI/NDWI/Cirrus 임계값 기반 cloud 검출
  - MY_LABELS     : 직접 수정 가능한 라벨 레이어 (저장 안 함, 테스트 전용)

임계값 파라미터는 CLI로 조정 가능.

사용 예:
  conda activate napari_env
  python test_segmentation.py --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2
  python test_segmentation.py --prepared_dir prepared/LC08_...  --bright 0.20 --cirrus 0.04
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


# ── 임계값 기반 cloud segmentation ────────────────────────────────────────

def threshold_cloud(bands: np.ndarray, fill_mask: np.ndarray,
                    bright_thr: float = 0.25,
                    cirrus_thr: float = 0.03,
                    ndsi_thr: float   = 0.40,
                    ndwi_thr: float   = 0.10,
                    bt_thr: float     = 280.0) -> np.ndarray:
    """
    밴드 임계값 기반 cloud 검출.

    bands : (H, W, 8) float32  — [B2, B3, B4, B5, B6, B7, B9, B10_BT]
             B2~B7, B9: TOA reflectance (0~1 범위)
             B10: Brightness Temperature (Kelvin)

    반환  : (H, W) uint8  — 라벨 번호 (label_scene.py 와 동일)
              0=미분류, 1=water, 2=snow, 3=shadow(미지원), 4=cloud, 255=fill
    """
    b2  = bands[:, :, 0]   # Blue
    b3  = bands[:, :, 1]   # Green
    b4  = bands[:, :, 2]   # Red
    b5  = bands[:, :, 3]   # NIR
    b6  = bands[:, :, 4]   # SWIR1
    b9  = bands[:, :, 6]   # Cirrus
    b10 = bands[:, :, 7]   # BT (K)

    # ── 지수 계산 ──────────────────────────────────────────────────────
    brightness = (b2 + b3 + b4) / 3.0

    denom_ndsi = b3 + b6
    denom_ndsi[denom_ndsi == 0] = 1e-6
    ndsi = (b3 - b6) / denom_ndsi          # > ndsi_thr → snow

    denom_ndwi = b3 + b5
    denom_ndwi[denom_ndwi == 0] = 1e-6
    ndwi = (b3 - b5) / denom_ndwi          # > ndwi_thr → water

    # ── 마스크 ──────────────────────────────────────────────────────────
    valid_bt   = b10 > 150                   # BT=0 은 fill → 무시
    cold_mask  = valid_bt & (b10 < bt_thr)  # cold pixel

    snow_mask  = (ndsi > ndsi_thr) & (b5 > 0.10)
    water_mask = (ndwi > ndwi_thr) & (brightness < 0.25)

    # cloud: (bright OR cold OR cirrus) AND NOT snow AND NOT water
    cloud_mask = (
        ((brightness > bright_thr) | cold_mask | (b9 > cirrus_thr))
        & ~snow_mask
        & ~water_mask
    )

    # ── 결과 조합 (우선순위: fill > cloud > snow > water) ──────────────
    result = np.zeros(bands.shape[:2], dtype=np.uint8)
    result[water_mask]  = 1
    result[snow_mask]   = 2
    result[cloud_mask]  = 4
    result[fill_mask]   = 255

    return result


# ── 씬 로드 ────────────────────────────────────────────────────────────────

def load_scene(prepared_dir: Path):
    meta = json.loads((prepared_dir / "meta.json").read_text())

    with rasterio.open(prepared_dir / "fci.tif") as src:
        fci = src.read()
    fci_rgb = np.transpose(fci, (1, 2, 0))   # (H, W, 3) uint8

    with rasterio.open(prepared_dir / "bands.tif") as src:
        bands = src.read()                    # (8, H, W) float32
    bands = np.transpose(bands, (1, 2, 0))   # (H, W, 8)

    with rasterio.open(prepared_dir / "cfmask.tif") as src:
        cfmask = src.read(1)                  # (H, W) uint8

    return fci_rgb, bands, cfmask, meta


# CFMask ref 레이어 고정 색상: 0=clear, 1=cloud, 2=shadow, 3=snow, 4=water
_CFMASK_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),    # clear  → 투명
    1: (1.0, 1.0, 1.0, 0.85),   # cloud  → 흰색
    2: (0.5, 0.1, 0.7, 0.85),   # shadow → 보라
    3: (0.0, 0.9, 0.9, 0.85),   # snow   → 하늘색
    4: (0.1, 0.3, 1.0, 0.85),   # water  → 파랑
}


# ── napari 실행 ────────────────────────────────────────────────────────────

def launch(prepared_dir: Path, bright_thr: float, cirrus_thr: float,
           ndsi_thr: float, ndwi_thr: float, bt_thr: float):
    import napari

    fci_rgb, bands, cfmask, meta = load_scene(prepared_dir)
    scene_id = meta["scene_id"]
    fill_mask = cfmask == 255

    print(f"[로드] {scene_id}  shape={fci_rgb.shape[:2]}")

    # ── threshold segmentation 계산 ─────────────────────────────────────
    thr_seg = threshold_cloud(bands, fill_mask,
                              bright_thr=bright_thr, cirrus_thr=cirrus_thr,
                              ndsi_thr=ndsi_thr, ndwi_thr=ndwi_thr, bt_thr=bt_thr)

    n_cloud_cf  = int((cfmask == 1).sum())
    n_cloud_thr = int((thr_seg == 4).sum())
    total       = cfmask.size
    print(f"[CFMask]    cloud: {n_cloud_cf / total * 100:.1f}%  ({n_cloud_cf:,} px)")
    print(f"[Threshold] cloud: {n_cloud_thr / total * 100:.1f}%  ({n_cloud_thr:,} px)")

    # ── napari ──────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"Segmentation Test — {scene_id}")
    viewer.add_image(fci_rgb, name="FCI (B7/B5/B3)", rgb=True)

    cfmask_display = cfmask.copy()
    cfmask_display[cfmask == 255] = 0
    cfmask_layer = viewer.add_labels(cfmask_display.astype(np.uint8),
                                     name="CFMask  ■흰=cloud ■하늘=snow ■보라=shadow ■파랑=water",
                                     opacity=0.40, visible=True)
    cfmask_layer.color = _CFMASK_COLORS

    viewer.add_labels(thr_seg,
                      name=f"Threshold (bright>{bright_thr} cirrus>{cirrus_thr} bt<{bt_thr}K)",
                      opacity=0.45, visible=True)

    # 직접 수정 가능한 빈 레이어
    H, W = fci_rgb.shape[:2]
    my_labels = np.zeros((H, W), dtype=np.uint8)
    my_labels[fill_mask] = 255
    viewer.add_labels(my_labels,
                      name="MY_LABELS (편집용, 저장 안 됨)",
                      opacity=0.50, visible=False)

    print("\n========================================")
    print("napari 비교 방법:")
    print("  - 레이어 눈 아이콘으로 CFMask / Threshold 토글 비교")
    print("  - Threshold 파라미터 변경: --bright / --cirrus / --bt 옵션")
    print("  - 4=cloud  2=snow  1=water  0=미분류  255=fill")
    print("========================================\n")

    napari.run()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임계값 기반 cloud segmentation 테스트 (napari)")
    parser.add_argument("--prepared_dir", type=Path, required=True,
                        help="prepare_scene.py 출력 디렉토리")
    parser.add_argument("--bright",  type=float, default=0.25,
                        help="가시광 밝기 임계값 (기본 0.25)")
    parser.add_argument("--cirrus",  type=float, default=0.03,
                        help="B9 Cirrus 임계값 (기본 0.03)")
    parser.add_argument("--ndsi",    type=float, default=0.40,
                        help="NDSI snow 임계값 (기본 0.40)")
    parser.add_argument("--ndwi",    type=float, default=0.10,
                        help="NDWI water 임계값 (기본 0.10)")
    parser.add_argument("--bt",      type=float, default=280.0,
                        help="B10 BT cold 임계값 K (기본 280K)")
    args = parser.parse_args()

    launch(args.prepared_dir,
           bright_thr=args.bright, cirrus_thr=args.cirrus,
           ndsi_thr=args.ndsi, ndwi_thr=args.ndwi, bt_thr=args.bt)
