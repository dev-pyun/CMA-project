"""
Landsat 8 OLI L1 씬을 라벨링용으로 준비.

입력  : Landsat L1 씬 폴더 (B2~B7, B9, B10, QA_PIXEL, MTL.txt)
출력  : prepared/<scene_id>/
          - bands.tif    : (8, H, W) float32, TOA reflectance + BT [B2,B3,B4,B5,B6,B7,B9,B10]
          - fci.tif      : (3, H, W) uint8,   FCI (B7/B5/B3 false color, gamma 0.5)
          - cfmask.tif   : (H, W)    uint8,   CFMask 5-class
          - meta.json    : scene_id, CRS, transform, sun angle 등

CFMask 클래스 코드:
  0: clear (non-special)
  1: cloud (Bit 1 dilated + Bit 3 cloud + Bit 2 cirrus)
  2: cloud shadow
  3: snow/ice
  4: water
  255: fill (Bit 0)

사용 예:
  python prepare_scene.py \\
      --scene_dir /home/immj/Landsat/USGS/.../LC08_L1GT_188114_20201114_..._02_T2 \\
      --out_dir   /home/immj/Labmeeting/project_cloud/data/prepared
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


# %% [1] MTL.txt 파싱 — 라디오메트릭 변환 계수 추출
def parse_mtl(mtl_path: Path) -> dict:
    """
    Landsat 8 MTL.txt 에서 reflectance/radiance/thermal 계수 추출.
    """
    text = mtl_path.read_text()

    def grab(key: str) -> float:
        m = re.search(rf"\s{re.escape(key)}\s*=\s*([\-0-9eE\.\+]+)", text)
        if m is None:
            raise KeyError(f"{key} not found in {mtl_path}")
        return float(m.group(1))

    coef = {"refl_mult": {}, "refl_add": {}, "rad_mult": {}, "rad_add": {}, "k1": {}, "k2": {}}

    # 반사율 변환 계수 (B1~B9)
    for b in range(1, 10):
        coef["refl_mult"][b] = grab(f"REFLECTANCE_MULT_BAND_{b}")
        coef["refl_add"][b] = grab(f"REFLECTANCE_ADD_BAND_{b}")

    # 라디언스 변환 계수 + 열적외선 상수 (B10, B11)
    for b in (10, 11):
        coef["rad_mult"][b] = grab(f"RADIANCE_MULT_BAND_{b}")
        coef["rad_add"][b] = grab(f"RADIANCE_ADD_BAND_{b}")
        coef["k1"][b] = grab(f"K1_CONSTANT_BAND_{b}")
        coef["k2"][b] = grab(f"K2_CONSTANT_BAND_{b}")

    coef["sun_elevation_deg"] = grab("SUN_ELEVATION")
    coef["sun_azimuth_deg"] = grab("SUN_AZIMUTH")
    return coef


# %% [2] DN → TOA reflectance / brightness temperature
def dn_to_toa_reflectance(dn: np.ndarray, mult: float, add: float, sun_elev_deg: float) -> np.ndarray:
    """
    DN → TOA Reflectance (sun angle 보정 포함).
    """
    rho = mult * dn.astype(np.float32) + add
    sin_elev = np.sin(np.deg2rad(sun_elev_deg))
    rho = rho / sin_elev
    rho = np.clip(rho, 0.0, 1.5).astype(np.float32)
    return rho


def dn_to_brightness_temp(dn: np.ndarray, mult: float, add: float, k1: float, k2: float) -> np.ndarray:
    """
    DN → Spectral Radiance → Brightness Temperature (Kelvin).
    """
    radiance = mult * dn.astype(np.float32) + add
    # log(0) 방지
    radiance = np.where(radiance > 0, radiance, np.nan).astype(np.float32)
    bt = k2 / np.log(k1 / radiance + 1.0)
    return bt.astype(np.float32)


# %% [3] QA_PIXEL → CFMask 5-class
def decode_qa_pixel(qa: np.ndarray) -> np.ndarray:
    """
    Landsat 8 Collection 2 QA_PIXEL 비트마스크 → 5-class CFMask.
    """
    fill = (qa & (1 << 0)).astype(bool)
    dilated_cloud = (qa & (1 << 1)).astype(bool)
    cirrus = (qa & (1 << 2)).astype(bool)
    cloud = (qa & (1 << 3)).astype(bool)
    cloud_shadow = (qa & (1 << 4)).astype(bool)
    snow = (qa & (1 << 5)).astype(bool)
    water = (qa & (1 << 7)).astype(bool)

    cfmask = np.zeros(qa.shape, dtype=np.uint8)  # 0 = clear (non-special)
    cfmask[water] = 4
    cfmask[snow] = 3
    cfmask[cloud_shadow] = 2
    cfmask[cloud | dilated_cloud | cirrus] = 1  # cloud 우선순위 높음
    cfmask[fill] = 255
    return cfmask


# %% [4] FCI 생성 — B7/B5/B3 false color, gamma 0.5
def make_fci(b7: np.ndarray, b5: np.ndarray, b3: np.ndarray, fill_mask: np.ndarray) -> np.ndarray:
    """
    FCI (False Color Image) — snow/cloud 시각 분리에 최적.
    R = B7 (SWIR2), G = B5 (NIR), B = B3 (Green).
    감마 0.5 (sqrt) 스트레치, fill 영역은 검정.
    """
    rgb = np.stack([b7, b5, b3], axis=-1)  # (H, W, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.sqrt(rgb)  # gamma=0.5
    rgb = (rgb * 255).astype(np.uint8)
    rgb[fill_mask] = 0  # fill 영역 검정
    return rgb  # (H, W, 3) uint8


# %% [5] 메인 — 씬 한 장 처리
def prepare_scene(scene_dir: Path, out_root: Path) -> Path:
    scene_id = scene_dir.name
    print(f"[준비 시작] {scene_id}")

    # MTL 찾기
    mtl_files = list(scene_dir.glob("*_MTL.txt"))
    if not mtl_files:
        raise FileNotFoundError(f"{scene_dir} 에 MTL.txt 없음")
    coef = parse_mtl(mtl_files[0])
    print(f"  Sun elevation: {coef['sun_elevation_deg']:.2f}°")

    # 출력 디렉토리
    out_dir = out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 밴드 파일 경로
    band_files = {b: scene_dir / f"{scene_id}_B{b}.TIF" for b in [2, 3, 4, 5, 6, 7, 9, 10]}
    qa_file = scene_dir / f"{scene_id}_QA_PIXEL.TIF"
    for b, f in band_files.items():
        if not f.exists():
            raise FileNotFoundError(f"B{b} 없음: {f}")
    if not qa_file.exists():
        raise FileNotFoundError(f"QA_PIXEL 없음: {qa_file}")

    # 첫 밴드로 메타 추출
    with rasterio.open(band_files[2]) as src:
        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform
        profile_ref = src.profile.copy()
    print(f"  Scene shape: ({H}, {W})  CRS: {crs}")

    # B2~B7, B9 → TOA reflectance, B10 → BT
    print("  밴드 변환 중...")
    bands_out = np.zeros((8, H, W), dtype=np.float32)  # ch 0~7
    band_index = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7}

    for b, ch in band_index.items():
        with rasterio.open(band_files[b]) as src:
            dn = src.read(1)
        if b <= 9:
            bands_out[ch] = dn_to_toa_reflectance(
                dn, coef["refl_mult"][b], coef["refl_add"][b], coef["sun_elevation_deg"]
            )
        else:  # B10
            bands_out[ch] = dn_to_brightness_temp(
                dn, coef["rad_mult"][b], coef["rad_add"][b], coef["k1"][b], coef["k2"][b]
            )
        print(f"    B{b} → ch{ch}  range=[{bands_out[ch].min():.3f}, {bands_out[ch].max():.3f}]")

    # QA_PIXEL → CFMask
    print("  CFMask 디코딩...")
    with rasterio.open(qa_file) as src:
        qa = src.read(1)
    cfmask = decode_qa_pixel(qa)
    fill_mask = cfmask == 255
    fill_pct = fill_mask.mean() * 100
    print(f"    Fill: {fill_pct:.1f}%  Cloud: {(cfmask==1).mean()*100:.1f}%  "
          f"Shadow: {(cfmask==2).mean()*100:.1f}%  Snow: {(cfmask==3).mean()*100:.1f}%  "
          f"Water: {(cfmask==4).mean()*100:.1f}%")

    # FCI 생성 (B7/B5/B3 ch 5/3/1)
    print("  FCI 생성...")
    fci = make_fci(bands_out[5], bands_out[3], bands_out[1], fill_mask)

    # 저장 (모두 GeoTIFF, CRS 보존)
    print(f"  저장 중: {out_dir}")

    # bands.tif
    profile_bands = profile_ref.copy()
    profile_bands.update(count=8, dtype="float32", compress="deflate", predictor=2)
    with rasterio.open(out_dir / "bands.tif", "w", **profile_bands) as dst:
        dst.write(bands_out)
        dst.descriptions = tuple([f"B{b}" for b in [2, 3, 4, 5, 6, 7, 9, 10]])

    # fci.tif (RGB)
    profile_fci = profile_ref.copy()
    profile_fci.update(count=3, dtype="uint8", compress="deflate", photometric="rgb")
    with rasterio.open(out_dir / "fci.tif", "w", **profile_fci) as dst:
        # rasterio는 (count, H, W) 순서, 내 fci는 (H, W, 3) → 전치
        dst.write(np.transpose(fci, (2, 0, 1)))

    # cfmask.tif
    profile_cf = profile_ref.copy()
    profile_cf.update(count=1, dtype="uint8", compress="deflate", nodata=255)
    with rasterio.open(out_dir / "cfmask.tif", "w", **profile_cf) as dst:
        dst.write(cfmask, 1)

    # meta.json
    meta = {
        "scene_id": scene_id,
        "shape_HW": [int(H), int(W)],
        "crs": str(crs),
        "transform": list(transform)[:6],
        "sun_elevation_deg": coef["sun_elevation_deg"],
        "sun_azimuth_deg": coef["sun_azimuth_deg"],
        "bands_order": ["B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10"],
        "fci_bands": ["B7_R", "B5_G", "B3_B"],
        "cfmask_classes": {
            "0": "clear",
            "1": "cloud",
            "2": "shadow",
            "3": "snow",
            "4": "water",
            "255": "fill",
        },
        "cfmask_pct": {
            "fill": float(fill_pct),
            "cloud": float((cfmask == 1).mean() * 100),
            "shadow": float((cfmask == 2).mean() * 100),
            "snow": float((cfmask == 3).mean() * 100),
            "water": float((cfmask == 4).mean() * 100),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[완료] {out_dir}\n")
    return out_dir


# %% [6] CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landsat 8 씬을 라벨링용으로 준비")
    parser.add_argument("--scene_dir", type=Path, required=True, help="Landsat L1 씬 폴더")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/immj/Labmeeting/project_cloud/data/prepared"),
        help="prepared 출력 루트 디렉토리",
    )
    args = parser.parse_args()

    prepare_scene(args.scene_dir, args.out_dir)
