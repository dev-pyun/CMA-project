# Validation / Test 씬 라벨링 워크플로우

---

## 환경 정리

| 단계 | 스크립트 | conda env | X11 필요 |
|------|----------|-----------|---------|
| 1. 씬 준비 | `prepare_scene.py` | `napari_env` | No |
| 2. napari 라벨링 | `label_scene.py` | `napari_env` | **Yes** |
| 3. Zarr 패치 생성 | `scene_to_patches.py` | `remote` | No |

---

## Step 1. 씬 준비 (`prepare_scene.py`)

```bash
conda activate napari_env
cd /home/pyuncb/src/label_code

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

# ── Validation ──────────────────────────────────────────────────────────
python prepare_scene.py --scene_dir $WEDDELL/2020/01/20200128/LC08_L1GT_199110_20200128_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/02/20200219/LC08_L1GT_177110_20200219_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/02/20200225/LC08_L1GT_171110_20200225_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/03/20200302/LC08_L1GT_165110_20200302_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/04/20200419/LC08_L1GT_181098_20200419_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/10/20201017/LC08_L1GT_200111_20201017_20201105_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/12/20201213/LC08_L1GT_199105_20201213_20210314_02_T2 --out_dir prepared/

# ── Test ────────────────────────────────────────────────────────────────
python prepare_scene.py --scene_dir $WEDDELL/2020/01/20200124/LC08_L1GT_187116_20200124_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/02/20200203/LC08_L1GT_177110_20200203_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/03/20200308/LC08_L1GT_175109_20200308_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/03/20200307/LC08_L1GT_184109_20200307_20201016_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/10/20201018/LC08_L1GT_207105_20201018_20201105_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/12/20201201/LC08_L1GT_195110_20201201_20210312_02_T2 --out_dir prepared/
python prepare_scene.py --scene_dir $WEDDELL/2020/12/20201215/LC08_L1GT_181114_20201215_20210314_02_T2 --out_dir prepared/
```

출력: `prepared/<scene_id>/` — `bands.tif`, `fci.tif`, `cfmask.tif`, `meta.json`

---

## Step 2. napari 라벨링 (`label_scene.py`)

> MobaXterm X11 포워딩 필요. 씬 하나씩 실행 후 창 닫으면 자동 저장.
>
> `--init_cfmask` 플래그: MY_LABELS 레이어를 CFMask 결과로 초기화 → 오탐·미탐 영역만 페인트하면 됨.

```bash
conda activate napari_env
cd /home/pyuncb/src/label_code

# ── Validation ──────────────────────────────────────────────────────────
# 188114 는 이미 완료 (V)
python label_scene.py --prepared_dir prepared/LC08_L1GT_199110_20200128_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_177110_20200219_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_171110_20200225_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_165110_20200302_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_181098_20200419_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_200111_20201017_20201105_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_199105_20201213_20210314_02_T2 --init_cfmask

# ── Test ────────────────────────────────────────────────────────────────
python label_scene.py --prepared_dir prepared/LC08_L1GT_187116_20200124_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_177110_20200203_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_175109_20200308_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_184109_20200307_20201016_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_207105_20201018_20201105_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_195110_20201201_20210312_02_T2 --init_cfmask
python label_scene.py --prepared_dir prepared/LC08_L1GT_181114_20201215_20210314_02_T2 --init_cfmask
```

저장 위치: `labels/<scene_id>_labels.tif`

**라벨링 키:**

| 키 | 클래스 | patch 저장값 |
|----|--------|------------|
| `4` | cloud (opaque + cirrus + dilated) | 1 |
| `3` | cloud shadow (명확한 경우만) | 1 |
| `2` | snow / ice | 0 |
| `1` | water | 0 |
| `0` | 미라벨 (지우기) | 255 (ignore) |

---

## Step 3. Zarr 패치 생성 (`scene_to_patches.py`)

```bash
conda activate remote
cd /home/pyuncb/src/label_code

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

# ── Validation → data/VALIDATION_ZARR/ ─────────────────────────────────
python scene_to_patches.py --scene_dir $WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 --label_path labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/01/20200128/LC08_L1GT_199110_20200128_20201016_02_T2 --label_path labels/LC08_L1GT_199110_20200128_20201016_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/02/20200219/LC08_L1GT_177110_20200219_20201016_02_T2 --label_path labels/LC08_L1GT_177110_20200219_20201016_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/02/20200225/LC08_L1GT_171110_20200225_20201016_02_T2 --label_path labels/LC08_L1GT_171110_20200225_20201016_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/03/20200302/LC08_L1GT_165110_20200302_20201016_02_T2 --label_path labels/LC08_L1GT_165110_20200302_20201016_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/04/20200419/LC08_L1GT_181098_20200419_20201016_02_T2 --label_path labels/LC08_L1GT_181098_20200419_20201016_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/10/20201017/LC08_L1GT_200111_20201017_20201105_02_T2 --label_path labels/LC08_L1GT_200111_20201017_20201105_02_T2_labels.tif --split val
python scene_to_patches.py --scene_dir $WEDDELL/2020/12/20201213/LC08_L1GT_199105_20201213_20210314_02_T2 --label_path labels/LC08_L1GT_199105_20201213_20210314_02_T2_labels.tif --split val

# ── Test → data/TEST_ZARR/ ──────────────────────────────────────────────
python scene_to_patches.py --scene_dir $WEDDELL/2020/01/20200124/LC08_L1GT_187116_20200124_20201016_02_T2 --label_path labels/LC08_L1GT_187116_20200124_20201016_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/02/20200203/LC08_L1GT_177110_20200203_20201016_02_T2 --label_path labels/LC08_L1GT_177110_20200203_20201016_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/03/20200308/LC08_L1GT_175109_20200308_20201016_02_T2 --label_path labels/LC08_L1GT_175109_20200308_20201016_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/03/20200307/LC08_L1GT_184109_20200307_20201016_02_T2 --label_path labels/LC08_L1GT_184109_20200307_20201016_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/10/20201018/LC08_L1GT_207105_20201018_20201105_02_T2 --label_path labels/LC08_L1GT_207105_20201018_20201105_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/12/20201201/LC08_L1GT_195110_20201201_20210312_02_T2 --label_path labels/LC08_L1GT_195110_20201201_20210312_02_T2_labels.tif --split test
python scene_to_patches.py --scene_dir $WEDDELL/2020/12/20201215/LC08_L1GT_181114_20201215_20210314_02_T2 --label_path labels/LC08_L1GT_181114_20201215_20210314_02_T2_labels.tif --split test
```
