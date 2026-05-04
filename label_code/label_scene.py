"""
napari 로 Landsat 씬 라벨링.

입력  : prepared/<scene_id>/ (prepare_scene.py 출력)
            - bands.tif, fci.tif, cfmask.tif, meta.json
출력  : labels/<scene_id>_labels.tif (uint8, GeoTIFF, CRS 보존)
            클래스:
                0   = 미라벨 (napari 기본값, patch 저장 시 255(ignore)로 remap)
                1   = clear land
                2   = water
                3   = snow / ice
                4   = cloud shadow  (명확히 보이는 경우만)
                5   = cloud  (opaque + thin cirrus + dilated 포함)
                255 = 센서 fill (자동 마킹)

            ※ 애매한 shadow / 경계 픽셀은 0(미라벨)으로 두면 ignore 처리됩니다.

학습 파이프라인 remap (scene_to_patches.py):
    {4, 5} → 1 (cloud)
    {1, 2, 3} → 0 (no-cloud)
    {0, 255} → 255 (ignore)

워크플로우:
  1. python label_scene.py --prepared_dir <prepared/scene_id>
  2. napari GUI 열림. 단축키:
       5 키     : cloud (opaque + cirrus + dilated 모두)
       4 키     : cloud shadow (명확한 경우만)
       3 키     : snow / ice
       2 키     : water
       1 키     : clear land
       0 키     : 미라벨로 초기화 (지우기)
       P 키     : Polygon mode (외곽선 클릭, 우클릭으로 종료)
       N 키     : Paint mode (브러시 칠하기)
       E 키     : Erase mode (지우기)
  3. 창 닫으면 자동 저장.

SAM (선택):
  --use_sam 플래그를 주면 napari-sam plugin 활성화 (GPU 자동감지)
  설치: pip install napari-sam segment-anything

사용 예:
  python label_scene.py \\
      --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \\
      --use_sam
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


# %% [1] 라벨 클래스 정의
# NOTE: 0 을 기본값(미라벨)으로 두는 이유 — napari 레이어 초기화 값이 0이므로
#       그림을 그리기 전 전체 픽셀이 0(미라벨)으로 시작. patch 저장 시 0 → 255(ignore) remap.
LABEL_CLASSES = {
    0:   ("nodata",  "미라벨 — 그리지 않은 영역 (patch 저장 시 255로 remap)"),
    1:   ("clear",   "clear land"),
    2:   ("water",   "water"),
    3:   ("snow",    "snow / ice"),
    4:   ("shadow",  "cloud shadow (명확한 경우만; 애매하면 0으로 두기)"),
    5:   ("cloud",   "cloud (opaque + thin cirrus + dilated 포함)"),
    255: ("fill",    "센서 결손 fill — 항상 ignore"),
}

NUM_LABEL_CLASSES = 6  # 0~5 (255 제외)


# %% [2] prepared 데이터 로드
def load_prepared(prepared_dir: Path):
    meta = json.loads((prepared_dir / "meta.json").read_text())
    print(f"[로드] {meta['scene_id']}")
    print(f"  shape: {meta['shape_HW']}  sun_elev: {meta['sun_elevation_deg']:.1f}°")

    with rasterio.open(prepared_dir / "fci.tif") as src:
        fci = src.read()          # (3, H, W)
        fci_profile = src.profile.copy()
    fci_rgb = np.transpose(fci, (1, 2, 0))  # (H, W, 3)

    with rasterio.open(prepared_dir / "cfmask.tif") as src:
        cfmask = src.read(1)

    return fci_rgb, cfmask, meta, fci_profile


# %% [3] napari 라벨링 GUI
def launch_napari(fci_rgb: np.ndarray, cfmask: np.ndarray, scene_id: str,
                  init_labels: np.ndarray = None, use_sam: bool = False):
    import napari

    viewer = napari.Viewer(title=f"Cloud Labeling — {scene_id}")

    # FCI background
    viewer.add_image(fci_rgb, name="FCI (B7/B5/B3)", rgb=True)

    # CFMask overlay (반투명, 위치 참고용)
    cfmask_display = cfmask.copy()
    cfmask_display[cfmask == 255] = 0
    viewer.add_labels(
        cfmask_display.astype(np.uint8),
        name="CFMask ref (1=cloud 2=shadow 3=snow 4=water)",
        opacity=0.35,
    )

    # 라벨 layer
    H, W = fci_rgb.shape[:2]
    if init_labels is None:
        labels_arr = np.zeros((H, W), dtype=np.uint8)
    else:
        labels_arr = init_labels.copy()
    labels_arr[cfmask == 255] = 255  # fill 영역 자동 마킹

    labels_layer = viewer.add_labels(
        labels_arr,
        name="MY_LABELS  (5=cloud  4=shadow  3=snow  2=water  1=clear  0=미라벨  255=fill)",
    )

    print("\n========================================")
    print("napari 라벨링 단축키:")
    print("  5        : cloud  (opaque + cirrus + dilated 모두)")
    print("  4        : cloud shadow  (명확한 경우만!)")
    print("  3        : snow / ice")
    print("  2        : water")
    print("  1        : clear land")
    print("  0        : 미라벨로 초기화 (지우기)")
    print("  P        : Polygon mode  (좌클릭→꼭짓점, 우클릭→종료)")
    print("  N        : Paint mode  (브러시)")
    print("  E        : Erase mode")
    print("  Space    : Pan  /  Z drag: Zoom")
    print("")
    print("  ※ 애매한 shadow·경계 픽셀은 0(미라벨)으로 두세요 → ignore 처리됩니다.")
    print("  ※ CFMask overlay를 참고해 shadow 위치를 추정하세요.")
    print("  ※ 창 닫으면 자동 저장됩니다.")
    print("========================================\n")

    if use_sam:
        try:
            import napari_sam  # noqa: F401
            print("[SAM] napari-sam 사용 가능. Plugins 메뉴에서 'Segment Anything' 활성화.")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SAM] device: {device}")
            if device == "cpu":
                print("[SAM] 경고: GPU 없음 → SAM 느림. vit_b 체크포인트 권장.")
        except ImportError:
            print("[SAM] napari-sam 미설치. SAM 없이 진행.")

    napari.run()

    return labels_layer.data.astype(np.uint8)


# %% [4] 라벨 저장 (GeoTIFF, CRS 보존)
def save_labels(labels: np.ndarray, prepared_dir: Path, out_path: Path):
    with rasterio.open(prepared_dir / "bands.tif") as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=255)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labels, 1)
    print(f"[저장] {out_path}")

    print("[통계]")
    for cls, (name, _) in LABEL_CLASSES.items():
        pct = (labels == cls).mean() * 100
        if pct > 0:
            print(f"  {cls:3d}  {name:8s}: {pct:6.2f}%")


# %% [5] CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="napari 로 Landsat 씬 라벨링")
    parser.add_argument("--prepared_dir", type=Path, required=True,
                        help="prepare_scene.py 가 만든 prepared/<scene_id> 폴더")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/home/pyuncb/src/label_code/labels"),
                        help="라벨 저장 루트 디렉토리")
    parser.add_argument("--use_sam", action="store_true", help="napari-sam plugin 활성화")
    parser.add_argument("--resume", action="store_true",
                        help="기존 라벨 있으면 이어서 작업")
    args = parser.parse_args()

    fci_rgb, cfmask, meta, _ = load_prepared(args.prepared_dir)
    scene_id = meta["scene_id"]
    out_path = args.out_dir / f"{scene_id}_labels.tif"

    init_labels = None
    if args.resume and out_path.exists():
        print(f"[resume] 기존 라벨 로드: {out_path}")
        with rasterio.open(out_path) as src:
            init_labels = src.read(1)

    final_labels = launch_napari(fci_rgb, cfmask, scene_id,
                                 init_labels=init_labels, use_sam=args.use_sam)
    save_labels(final_labels, args.prepared_dir, out_path)
    print("\n[완료] 다음 단계: scene_to_patches.py 로 256×256 patch 분할.")
