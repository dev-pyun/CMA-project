"""
라벨된 Landsat 씬 → 256×256 patch 자동 분할.

입력  : prepared/<scene_id>/bands.tif  (8, H, W) float32
        labels/<scene_id>_labels.tif   (H, W) uint8
            label_scene.py 출력 scheme:
                0   = 미라벨 (napari 기본값)
                1   = water
                2   = snow / ice
                3   = cloud shadow
                4   = cloud (opaque + cirrus + dilated)
                255 = 센서 fill

출력  : patches/<split>/<scene_id>_p{i:05d}_{j:05d}.h5
            /input  (8, 256, 256) float32
            /label  (256, 256)    uint8
                remap 후:
                    0   = no-cloud  (원본 1/2)
                    1   = cloud     (원본 3/4)
                    255 = ignore    (원본 0/255)
            attrs   : scene_id, row/col, valid_label_frac,
                      has_cloud, has_shadow, has_snow, has_water, cloud_frac, ...

필터링 규칙:
  - fill 비율 > 50%         → 버림
  - 유효 라벨 비율 < 5%    → 'train_aux'
  - 유효 라벨 비율 ≥ 5%    → 'val'

사용 예:
  python scene_to_patches.py \\
      --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \\
      --label_path   labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \\
      --out_root     patches/ \\
      --patch_size 256 --stride 256
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm


# label_scene.py 출력 → 학습 파이프라인 binary 형식
# {shadow(3), cloud(4)} → 1(cloud)
# {water(1), snow(2)} → 0(no-cloud)
# {nodata(0), fill(255)} → 255(ignore)
LABEL_REMAP = {
    0:   255,
    1:   0,
    2:   0,
    3:   1,
    4:   1,
    255: 255,
}

# 유효 라벨로 인정되는 원본 값 (미라벨·fill 제외)
VALID_LABEL_VALUES = {1, 2, 3, 4}


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """label_scene.py 출력(0/1/2/3/4/5/255) → 학습 형식(0/1/255)."""
    out = np.full_like(labels, 255, dtype=np.uint8)
    for src_val, dst_val in LABEL_REMAP.items():
        out[labels == src_val] = dst_val
    return out


# %% [1] Patch 자르기 + 필터링
def split_into_patches(
    bands: np.ndarray,
    labels_raw: np.ndarray,
    cfmask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_label_frac: float = 0.05,
    max_fill_frac: float = 0.5,
):
    """
    bands      : (C, H, W) float32
    labels_raw : (H, W) uint8  — label_scene.py 출력 (0~5 / 255)
    cfmask     : (H, W) uint8  — fill 식별용

    yield (input_patch, label_patch_remapped, attrs_dict, split_name)
    """
    C, H, W = bands.shape
    assert labels_raw.shape == (H, W), "labels shape mismatch"

    n_total, n_kept_val, n_kept_aux, n_skipped = 0, 0, 0, 0

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            n_total += 1
            b  = bands[:, i:i + patch_size, j:j + patch_size]
            lr = labels_raw[i:i + patch_size, j:j + patch_size]
            cf = cfmask[i:i + patch_size, j:j + patch_size]

            fill_frac = (cf == 255).mean()
            if fill_frac > max_fill_frac:
                n_skipped += 1
                continue

            # 유효 라벨: 1~4 (미라벨 0 · fill 255 제외)
            valid_mask = np.isin(lr, list(VALID_LABEL_VALUES))
            valid_frac = valid_mask.mean()

            attrs = {
                "row_start":          int(i),
                "col_start":          int(j),
                "fill_frac":          float(fill_frac),
                "valid_label_frac":   float(valid_frac),
                "has_cloud":          bool((lr == 4).any()),
                "has_shadow":         bool((lr == 3).any()),
                "has_snow":           bool((lr == 2).any()),
                "has_water":          bool((lr == 1).any()),
                "cloud_frac":         float((lr == 4).mean()),
                "shadow_frac":        float((lr == 3).mean()),
                "snow_frac":          float((lr == 2).mean()),
                "water_frac":         float((lr == 1).mean()),
                "cfmask_cloud_frac":  float((cf == 1).mean()),
                "cfmask_shadow_frac": float((cf == 2).mean()),
                "cfmask_snow_frac":   float((cf == 3).mean()),
                "cfmask_water_frac":  float((cf == 4).mean()),
            }

            split = "val" if valid_frac >= min_valid_label_frac else "train_aux"
            if split == "val":
                n_kept_val += 1
            else:
                n_kept_aux += 1

            yield b, remap_labels(lr), attrs, split

    print(f"\n[patch 통계]")
    print(f"  총 {n_total} patches 검토")
    print(f"  val      (유효 라벨 ≥ {min_valid_label_frac*100:.0f}%) : {n_kept_val}")
    print(f"  train_aux(유효 라벨 < {min_valid_label_frac*100:.0f}%) : {n_kept_aux}")
    print(f"  skipped  (fill > {max_fill_frac*100:.0f}%)             : {n_skipped}")


# %% [2] HDF5 저장
def save_patch_h5(path: Path, input_patch: np.ndarray, label_patch: np.ndarray,
                  attrs: dict, scene_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("input", data=input_patch.astype(np.float16),
                         compression="gzip", compression_opts=4)
        f.create_dataset("label", data=label_patch.astype(np.uint8),
                         compression="gzip")
        f.attrs["scene_id"] = scene_id
        for k, v in attrs.items():
            f.attrs[k] = v


# %% [3] 메인 — 한 씬 처리
def process_scene(
    prepared_dir: Path,
    label_path: Path,
    out_root: Path,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_label_frac: float = 0.05,
    max_fill_frac: float = 0.5,
):
    meta = json.loads((prepared_dir / "meta.json").read_text())
    scene_id = meta["scene_id"]
    print(f"[처리 시작] {scene_id}")

    with rasterio.open(prepared_dir / "bands.tif") as src:
        bands = src.read().astype(np.float32)
    print(f"  bands shape: {bands.shape}")

    with rasterio.open(prepared_dir / "cfmask.tif") as src:
        cfmask = src.read(1).astype(np.uint8)

    with rasterio.open(label_path) as src:
        labels_raw = src.read(1).astype(np.uint8)

    print(f"  labels 원본 통계:")
    names = {0:"미라벨", 1:"water", 2:"snow", 3:"shadow", 4:"cloud", 255:"fill"}
    for v, name in names.items():
        pct = (labels_raw == v).mean() * 100
        if pct > 0:
            print(f"    {v:3d} {name:8s}: {pct:.1f}%")

    saved = {"val": 0, "train_aux": 0}
    for input_patch, label_patch, attrs, split in tqdm(
        split_into_patches(bands, labels_raw, cfmask,
                           patch_size, stride, min_valid_label_frac, max_fill_frac),
        desc="patches",
    ):
        i, j = attrs["row_start"], attrs["col_start"]
        out_path = out_root / split / f"{scene_id}_p{i:05d}_{j:05d}.h5"
        save_patch_h5(out_path, input_patch, label_patch, attrs, scene_id)
        saved[split] += 1

    print(f"\n[저장 완료]")
    print(f"  remap: {{1,2}}→0(no-cloud)  {{3,4}}→1(cloud)  {{0,255}}→255(ignore)")
    for k, v in saved.items():
        print(f"  {k}: {v} patches → {out_root / k}")
    print(f"  총 {sum(saved.values())} patches\n")


# %% [4] CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="라벨된 씬 → 256×256 patch 분할")
    parser.add_argument("--prepared_dir",         type=Path, required=True)
    parser.add_argument("--label_path",           type=Path, required=True)
    parser.add_argument("--out_root",             type=Path,
                        default=Path("/home/pyuncb/src/label_code/patches"))
    parser.add_argument("--patch_size",           type=int,   default=256)
    parser.add_argument("--stride",               type=int,   default=256)
    parser.add_argument("--min_valid_label_frac", type=float, default=0.05)
    parser.add_argument("--max_fill_frac",        type=float, default=0.5)
    args = parser.parse_args()

    process_scene(
        prepared_dir=args.prepared_dir,
        label_path=args.label_path,
        out_root=args.out_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_label_frac=args.min_valid_label_frac,
        max_fill_frac=args.max_fill_frac,
    )
