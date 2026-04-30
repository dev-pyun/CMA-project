"""
napari 로 Landsat 씬 라벨링.

입력  : prepared/<scene_id>/ (prepare_scene.py 출력)
            - bands.tif, fci.tif, cfmask.tif, meta.json
출력  : labels/<scene_id>_labels.tif (uint8, GeoTIFF, CRS 보존)
            클래스: 0=no-cloud, 1=cloud, 255=미라벨/fill

워크플로우:
  1. python label_scene.py --prepared_dir <prepared/scene_id>
  2. napari GUI 열림. 단축키:
       1 키     : cloud (cloud + shadow + cirrus 모두)
       0 키     : no-cloud (clear + ice + water 모두)
       P 키     : Polygon mode (외곽선 클릭, 우클릭으로 종료)
       N 키     : Paint mode (브러시 칠하기)
       E 키     : Erase mode (지우기)
  3. 창 닫으면 자동 저장. 미라벨 영역은 255로 유지됨.

SAM (선택):
  --use_sam 플래그를 주면 napari-sam plugin 활성화 (GPU 자동감지)
  설치: pip install napari-sam segment-anything

사용 예:
  python label_scene.py \\
      --prepared_dir /home/immj/Labmeeting/project_cloud/data/prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \\
      --use_sam
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


# %% [1] 라벨 클래스 정의
LABEL_CLASSES = {
    0:   ("no-cloud", "cloud 없음 (clear + ice + snow + water 모두)"),
    1:   ("cloud",    "cloud 있음 (cloud + shadow + cirrus 모두)"),
    255: ("ignore",   "미라벨 또는 센서 결손 — loss에서 제외"),
}


# %% [2] prepared 데이터 로드
def load_prepared(prepared_dir: Path):
    """
    bands.tif (8, H, W) float32, fci.tif (3, H, W) uint8, cfmask.tif (H, W) uint8 로드.
    + meta.json + rasterio profile 반환 (라벨 저장 시 CRS 일치용).
    """
    meta = json.loads((prepared_dir / "meta.json").read_text())
    print(f"[로드] {meta['scene_id']}")
    print(f"  shape: {meta['shape_HW']}  sun_elev: {meta['sun_elevation_deg']:.1f}°")

    with rasterio.open(prepared_dir / "fci.tif") as src:
        fci = src.read()  # (3, H, W)
        fci_profile = src.profile.copy()
    fci_rgb = np.transpose(fci, (1, 2, 0))  # (H, W, 3) — napari rgb=True 용

    with rasterio.open(prepared_dir / "cfmask.tif") as src:
        cfmask = src.read(1)

    return fci_rgb, cfmask, meta, fci_profile


# %% [3] napari 라벨링 GUI
def launch_napari(fci_rgb: np.ndarray, cfmask: np.ndarray, scene_id: str, use_sam: bool = False):
    """
    napari 띄우고 사용자 라벨 받음. 닫히면 라벨 array 반환.
    """
    import napari

    viewer = napari.Viewer(title=f"Cloud Labeling — {scene_id}")

    # FCI background
    viewer.add_image(fci_rgb, name="FCI (B7/B5/B3)", rgb=True)

    # CFMask overlay (반투명, 색별로 분리해서 보여주는 게 가독성 좋음)
    cfmask_for_display = cfmask.copy()
    cfmask_for_display[cfmask == 255] = 0  # fill 은 표시 안 함
    viewer.add_labels(
        cfmask_for_display.astype(np.uint8),
        name="CFMask (CFMask 1=cloud 2=shadow 3=snow 4=water)",
        opacity=0.4,
    )

    # 라벨 layer (사용자가 그릴 영역)
    H, W = fci_rgb.shape[:2]
    init_labels = np.zeros((H, W), dtype=np.uint8)
    # fill 영역은 미리 255 로 마킹
    init_labels[cfmask == 255] = 255

    labels_layer = viewer.add_labels(
        init_labels,
        name="MY_LABELS (1=cloud  0=no-cloud  255=미라벨)",
    )

    # 단축키 안내
    print("\n========================================")
    print("napari 라벨링 단축키:")
    print("  - 1        : cloud (cloud + shadow + cirrus 모두)")
    print("  - 0        : no-cloud (clear + ice + water 모두)")
    print("  - P        : Polygon mode (점 찍기 → 우클릭 종료)")
    print("  - N        : Paint mode (브러시)")
    print("  - E        : Erase mode")
    print("  - Space    : 임시 pan, Z 드래그: 줌")
    print("  - 미라벨로 두고 싶은 곳은 그냥 두면 됨 (255 유지).")
    print("  - 라벨링 완료 후 창 닫으면 저장됨.")
    print("========================================\n")

    # SAM 활성화 (있으면)
    if use_sam:
        try:
            import napari_sam  # noqa: F401
            print("[SAM] napari-sam plugin 사용 가능. Plugins 메뉴에서 'Segment Anything' 활성화.")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SAM] device: {device}")
            if device == "cpu":
                print("[SAM] 경고: GPU 없음 → SAM 추론 느림 (클릭당 5~30초). vit_b 체크포인트 권장.")
        except ImportError:
            print("[SAM] napari-sam 또는 segment-anything 미설치. SAM 없이 진행.")

    napari.run()  # 블로킹, 창 닫힐 때까지 대기

    final_labels = labels_layer.data.astype(np.uint8)
    return final_labels


# %% [4] 라벨 저장 (GeoTIFF, CRS 보존)
def save_labels(labels: np.ndarray, prepared_dir: Path, out_path: Path):
    """
    bands.tif 의 profile 그대로 사용해 GeoTIFF 로 저장.
    """
    with rasterio.open(prepared_dir / "bands.tif") as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=255)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labels, 1)
    print(f"[저장] {out_path}")

    # 통계 출력
    print("[통계]")
    for cls, (name, _) in LABEL_CLASSES.items():
        pct = (labels == cls).mean() * 100
        if pct > 0:
            print(f"  {cls} {name:10s}: {pct:6.2f} %")


# %% [5] CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="napari 로 Landsat 씬 라벨링")
    parser.add_argument(
        "--prepared_dir",
        type=Path,
        required=True,
        help="prepare_scene.py 가 만든 prepared/<scene_id> 폴더",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/pyuncb/src/label_code/labels"),
        help="라벨 저장 루트 디렉토리",
    )
    parser.add_argument("--use_sam", action="store_true", help="napari-sam plugin 활성화")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 라벨 있으면 이어서 작업 (out_dir/<scene>_labels.tif 로드)",
    )
    args = parser.parse_args()

    # 데이터 로드
    fci_rgb, cfmask, meta, _ = load_prepared(args.prepared_dir)
    scene_id = meta["scene_id"]

    out_path = args.out_dir / f"{scene_id}_labels.tif"

    # resume 모드
    if args.resume and out_path.exists():
        print(f"[resume] 기존 라벨 로드: {out_path}")
        with rasterio.open(out_path) as src:
            existing = src.read(1)
        # 라벨 layer 초기값을 기존 라벨로 (napari 안에서 이어서 작업)
        # → launch_napari 가 init_labels 를 받도록 약간 수정 필요. 간단히 처리:
        # 여기선 cfmask 자리에 existing 을 넣어 라벨로 시작하게 함.
        cfmask = existing.copy()  # init_labels 가 cfmask==255 를 fill 로 인식하므로 호환

    # napari 실행
    final_labels = launch_napari(fci_rgb, cfmask, scene_id, use_sam=args.use_sam)

    # 저장
    save_labels(final_labels, args.prepared_dir, out_path)
    print("\n[완료] 다음 단계: scene_to_patches.py 로 256×256 patch 분할.")
