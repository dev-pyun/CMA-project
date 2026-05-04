# Cloud Masking — 수동 라벨링 도구 사용법

남극 Landsat 8 cloud mask 검증용 수동 라벨링 파이프라인.
napari GUI로 각 씬을 직접 라벨링하고, 256×256 patch로 분할해 검증 데이터를 만든다.

---

## 클래스 코드 요약

### napari 라벨링 시 (label_scene.py 출력)

| Code | Class | 정의 | napari 키 |
|------|-------|------|----------|
| 0 | 미라벨 | 라벨링 안 한 영역 (napari 기본값) | `0` |
| 1 | clear | Clear land | `1` |
| 2 | water | Water | `2` |
| 3 | snow | Snow / Ice | `3` |
| 4 | shadow | Cloud shadow (**명확한 경우만**) | `4` |
| 5 | cloud | Cloud (opaque + thin cirrus + dilated 포함) | `5` |
| 255 | fill | 센서 결손 (자동 마킹) | — |

> **왜 0을 미라벨로?** napari 레이어 초기값이 0입니다. 0을 no-cloud로 쓰면 라벨링 전 전체 씬이 already labeled처럼 보이는 혼동이 생겨, 0을 미라벨(미작업)로 정의합니다.

> **Shadow 라벨링 원칙:** CFMask overlay + FCI 영상이 모두 어두운 경우만 4(shadow)로 라벨링. 경계가 불분명하거나 dark water / dark rock과 구분이 어려운 픽셀은 **0(미라벨)으로 두면 ignore 처리**됩니다.

### patch 저장 후 (scene_to_patches.py 자동 remap)

| 저장 값 | 의미 | 원본 클래스 | loss |
|---------|------|-----------|------|
| 0 | no-cloud | clear(1) + water(2) + snow(3) | 포함 |
| 1 | cloud | shadow(4) + cloud(5) | 포함 |
| 255 | ignore | 미라벨(0) + fill(255) | **무시** (ignore_index=255) |

remap 규칙: `{1,2,3}→0`, `{4,5}→1`, `{0,255}→255`

---

## 환경 설정

### 서버 환경

```bash
conda create -n cloud_label python=3.10 -y
conda activate cloud_label
pip install -r requirements.txt

# PyTorch CUDA (CUDA 버전 확인 후)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# SAM checkpoint (선택, vit_b 권장)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O ~/sam_vit_b.pth
```

### GUI 실행 환경 (Windows → 서버 X11 포워딩)

napari는 GUI 프로그램이므로 화면 출력이 필요합니다.

**MobaXterm 사용 (권장)**:
1. MobaXterm 설치: https://mobaxterm.mobatek.net/download.html
2. MobaXterm에서 SSH 세션 생성 → 서버 접속
3. X11 포워딩이 기본 활성화되어 있음 (별도 설정 불필요)
4. 접속 후 확인:
   ```bash
   echo $DISPLAY   # :0 또는 localhost:10.0 형태가 나와야 함
   ```

---

## 워크플로우 — 한 씬 라벨링

### Step 1. 씬 준비 (`prepare_scene.py`)

```bash
conda activate cloud_label
python prepare_scene.py \
    --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --out_dir   prepared/
```

출력 (`prepared/<scene_id>/`):

| 파일 | 내용 |
|------|------|
| `bands.tif` | (8, H, W) float32 — B2~B7+B9 TOA reflectance + B10 BT |
| `fci.tif` | (3, H, W) uint8 — FCI (B7/B5/B3) gamma 0.5 |
| `cfmask.tif` | (H, W) uint8 — CFMask 5-class |
| `meta.json` | CRS, sun angle, class 비율 등 |

### Step 2. napari 라벨링 (`label_scene.py`)

```bash
python label_scene.py \
    --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2
    # GPU 있으면: --use_sam
    # 이어서 작업: --resume
```

**napari 단축키:**

| 키 | 동작 |
|----|------|
| `5` | cloud 칠하기 (opaque + cirrus + dilated 모두) |
| `4` | cloud shadow 칠하기 (**명확한 경우만**) |
| `3` | snow / ice 칠하기 |
| `2` | water 칠하기 |
| `1` | clear land 칠하기 |
| `0` | 미라벨로 초기화 (지우기) |
| `P` | Polygon mode (좌클릭→꼭짓점, 우클릭→종료) |
| `N` | Paint mode (브러시) |
| `E` | Erase mode |
| `Space` | Pan (임시) |

**라벨링 원칙:**
- **확실한 영역만 라벨링** — 경계·애매한 픽셀은 0으로 두면 ignore 처리됨
- Shadow는 CFMask overlay + FCI 영상이 **모두 어두운 경우만** 라벨링
  - Dark water / dark rock과 구분 안 될 때 → 0(미라벨)으로 두기
  - Shadow 경계 2~3픽셀은 0으로 두어도 됨
- Cloud는 thin cirrus·dilated 포함해 5로 통일 (별도 구분 불필요)
- 창 닫으면 자동 저장 → `labels/<scene_id>_labels.tif`

**저장 후 통계 예시:**
```
[통계]
    0  nodata  : 45.23%   ← 미라벨 (라벨링 안 한 영역)
    1  clear   :  8.11%
    2  water   : 12.50%
    3  snow    : 15.20%
    4  shadow  :  2.30%
    5  cloud   : 16.41%
  255  fill    :  0.25%
```

### Step 3. patch 분할 (`scene_to_patches.py`)

```bash
python scene_to_patches.py \
    --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path   labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --out_root     patches/ \
    --patch_size 256 --stride 256
```

이 단계에서 자동 remap: `{1,2,3}→0`, `{4,5}→1`, `{0,255}→255(ignore)`

출력 (`patches/`):

| 경로 | 조건 | 용도 |
|------|------|------|
| `val/<scene>_p{i}_{j}.h5` | 유효 라벨 ≥ 5% | 검증 메인 |
| `train_aux/<scene>_p{i}_{j}.h5` | 유효 라벨 < 5% | 보조 학습용 |

각 patch HDF5:
- `/input` (8, 256, 256) float16 — 8-band 입력
- `/label` (256, 256) uint8 — remap 후 (0=no-cloud, 1=cloud, 255=ignore)
- `attrs` — scene_id, row/col, valid_label_frac, cloud/shadow/snow/water/clear_frac 등

---

## 참고

- Nambiar 논문: `/home/immj/Labmeeting/project_cloud/paper/Self-trained model for cloud, shadow and snow detection in sentinel-2 images of snow- and ice- covered regions.pdf`
