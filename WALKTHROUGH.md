# WALKTHROUGH — Landsat 8 Cloud Detection Pipeline

---

## [2026-04-30] 패치 포맷 전환 (HDF5 → Zarr) + Binary 라벨 + 파생 피처 추가

### 배경 및 목표
- QA_PIXEL 6-class 라벨을 **binary cloud / no-cloud**로 단순화
- 패치 저장 시 학습에 활용 가능한 모든 파생 피처(RGB, HSV, Sobel)를 미리 계산해 저장
- 저장 포맷을 HDF5 → **Zarr + blosc/zstd 압축**으로 교체 (랜덤 액세스 성능 및 압축률 개선)

---

### 수정 파일 목록

#### 1. `utils/qa_pixel_mapping.py`
- `qa_pixel_to_binary()` 함수 추가
  - Cloud 마스크: Bit 1 (Dilated) | Bit 2 (Cirrus) | Bit 3 (Cloud) | Bit 4 (Shadow) → **label 1**
  - No-Cloud: Clear, Snow, Water → **label 0**
  - Fill (Bit 0) → **label 255** (nodata, loss에서 ignore)
- 기존 `qa_pixel_to_classes()` (6-class)에 누락되어 있던 **Cirrus (Bit 2)** 처리 추가
- Binary 관련 상수 추가: `BINARY_NOCLOUD=0`, `BINARY_CLOUD=1`, `BINARY_NODATA=255`

#### 2. `utils/split_scene.py`
- **h5py 제거 → zarr + numcodecs 사용**
- 파생 피처 계산 함수 추가:
  - `compute_rgb()`: B4/B3/B2 퍼센타일 정규화 (2–98th) → float32 [0,1]
  - `compute_hsv()`: RGB → HSV via cv2, 정규화 → float32 [0,1]
  - `compute_sobel()`: 루미넌스 그레이스케일 기반 Sobel X, Y, Magnitude → float32
- `save_patch_zarr()`: zarr 그룹에 named arrays로 저장
  - `spectral` (H,W,8) uint16 — blosc/zstd bitshuffle
  - `rgb`      (H,W,3) float32 — blosc/zstd shuffle
  - `hsv`      (H,W,3) float32 — blosc/zstd shuffle
  - `sobel`    (H,W,3) float32 — blosc/zstd shuffle
  - `label`    (H,W)   uint8   — blosc/zstd bitshuffle
- 노데이터 픽셀(label==255)이 있는 패치 스킵

#### 3. `utils/dir_paths.py`
- `TRAIN_PATH` / `VALID_PATH` → `TRAIN_ZARR`, `VALIDATION_ZARR` 디렉토리로 변경
- 기존 HDF5 경로는 `TRAIN_H5_PATH` / `VALID_H5_PATH`로 보존 (하위 호환)

#### 4. `dataset/patch_dataset.py`
- **h5py 제거 → zarr 사용**
- 글로브 패턴 `*.h5` → `*.zarr`
- 패치 로딩: spectral + rgb + hsv + sobel 연결 → **(H,W,17) 텐서**
  - 채널 0–7: B1–B9 (`/10000` 정규화)
  - 채널 8–10: RGB (precomputed)
  - 채널 11–13: HSV (precomputed)
  - 채널 14–16: Sobel X, Y, Magnitude (precomputed)
- 패딩 시 label border = 255 (nodata)로 변경 (기존 0에서)
- `pseudo_label` 없이 stage 1+ 접근 시 명확한 RuntimeError

#### 5. `dataset/network_input.py`
- `BAND_INDEX`에 derived 채널 인덱스 추가:
  `RGB_R=8, RGB_G=9, RGB_B=10, HSV_H=11, HSV_S=12, HSV_V=13, Sobel_X=14, Sobel_Y=15, Sobel_Mag=16`
- 새 프리셋 추가:
  `rgb_precomp`, `hsv`, `sobel`, `rgb_hsv`, `swir_sobel`, `swirndsi_sobel`, `all_derived`

#### 6. `dataset/transforms.py`
- `CutOut`: 잘라낸 영역의 label을 `0` → **`255`**로 변경
  - 이전: 0으로 설정 → binary에서 "no-cloud"로 학습됨 (잘못된 동작)
  - 이후: 255로 설정 → `ignore_index=255`에 의해 loss에서 제외

#### 7. `network/model.py`
- `NUM_CLASSES = 2` (binary: 0=no-cloud, 1=cloud)
- `NODATA_LABEL = 255`
- `F.cross_entropy(..., ignore_index=255)` 추가 (nodata 픽셀 loss 제외)
- `Metrics(self.device, num_classes=NUM_CLASSES)` 명시적 전달
- `calculate_confusion_matrix(..., num_classes=NUM_CLASSES)` 명시적 전달
- **h5py 제거 → zarr 사용** (`generate_train_data`)
  - `pseudo_label` / `raw_prediction` 키로 zarr 배열 저장
  - nodata propagation: QA label의 255 위치 유지
- `encode_label()`: low-confidence 픽셀 → `0` → **`255`** (nodata로 처리)
- CSV 헤더: `NOCLOUD_F, CLOUD_F` (6-class에서 2-class로)

#### 8. `utils/metrics.py`
- `calculate_accuracy()`: `labels >= 0` → `(labels >= 0) & (labels < 255)` (nodata 제외)

#### 9. `utils/MFB.py`
- `calculate_file_freq()`: `label < num_classes` 마스크로 nodata(255) 제외 후 빈도 계산

#### 10. `inspect_h5.py`
- zarr 패치 지원으로 전면 재작성
- binary 라벨 시각화 (No-Cloud/Cloud/No-Data 3색)
- spectral + RGB + HSV + Sobel Magnitude + NDSI 패널 표시
- `.zgroup` 파일 유무로 단일 패치 / 상위 디렉토리 자동 구분

---

## [2026-04-30] label_code/README.md 라벨 스키마 통일

### 변경 이유
`label_code/README.md`의 클래스 코드가 기존 Nambiar 6-class 기준(0=unlabeled, 1=cloud, 2=shadow, 3=ice, 4=water, 255=fill)으로 작성되어 있었음. 현재 파이프라인의 binary 스키마와 불일치.

### 수정 파일
`label_code/label_scene.py`:
- `LABEL_CLASSES` dict: 4-class → binary (`0=no-cloud`, `1=cloud`, `255=ignore`)
- 레이어 이름: `"MY_LABELS (1=cloud 2=shadow 3=ice 4=water)"` → `"MY_LABELS (1=cloud 0=no-cloud 255=미라벨)"`
- 단축키 안내 출력: `숫자 1~4` → `0/1` binary 기준으로 변경
- 모듈 docstring 업데이트

`label_code/scene_to_patches.py`:
- `valid_label` 필터링: `(l > 0) & (l != 255)` → `(l != 255)` (0=no-cloud도 유효 라벨)
- `attrs` dict: `has_shadow`, `has_ice`, `has_water` 제거 → `has_nocloud`, `cloud_frac` 추가
- 통계 출력: cloud/shadow/ice/water 분리 → no-cloud/cloud/미라벨 binary 기준
- 모듈 docstring 업데이트

`label_code/README.md`:
- 클래스 코드 표 변경:
  - 이전: 0=unlabeled, 1=cloud, 2=shadow, 3=ice, 4=water, 255=fill
  - 이후: **0=no-cloud, 1=cloud, 255=미라벨/fill**
- 라벨링 단축키 변경:
  - 이전: `1`/`2`/`3`/`4` = cloud/shadow/ice/water
  - 이후: `1` = cloud(+shadow+cirrus), `0` = no-cloud(+ice+water)
- patch HDF5 `/label` 설명 업데이트: `0=no-cloud, 1=cloud, 255=미라벨/fill`
- 라벨링 원칙: unlabeled → 미라벨(255)로 표현 통일
- shadow oversample 항목 제거 (binary에서 불필요)

---

### 새 패치 생성 방법

```bash
# zarr 패치 생성 (기존 H5 패치와 별도로 TRAIN_ZARR/ 에 생성됨)
pip install zarr numcodecs
python make_landsat_data.py --mode train

# 패치 내용 확인
python inspect_h5.py data/TRAIN_ZARR/LC08_..._PATCH0.zarr
python inspect_h5.py data/TRAIN_ZARR/ --sample 6 --save
```

### 학습 방법 (변경 없음)

```bash
python train.py -e my_exp -st 0 -ip swirndsi -gpu 0

# 파생 피처 활용 예
python train.py -e my_exp -st 0 -ip swirndsi_sobel -gpu 0
python train.py -e my_exp -st 0 -ip all_derived -gpu 0
```

---

### 채널 레이아웃 (패치 내부)

| zarr 배열 | shape | dtype | 설명 |
|---|---|---|---|
| `spectral` | (256,256,8) | uint16 | B1–B7, B9 raw DN |
| `rgb` | (256,256,3) | float32 | B4/B3/B2 퍼센타일 정규화 [0,1] |
| `hsv` | (256,256,3) | float32 | H,S,V [0,1] |
| `sobel` | (256,256,3) | float32 | Sobel_X, Sobel_Y, Magnitude |
| `label` | (256,256) | uint8 | 0=no-cloud, 1=cloud, 255=nodata |
| `pseudo_label` | (256,256) | uint8 | stage 1+에서 label_generation.py가 추가 |

### 학습 텐서 채널 레이아웃 (17채널)

| 채널 | 이름 | 소스 |
|---|---|---|
| 0–7 | B1–B7, B9 | spectral/10000 |
| 8–10 | RGB_R, RGB_G, RGB_B | zarr rgb 배열 |
| 11–13 | HSV_H, HSV_S, HSV_V | zarr hsv 배열 |
| 14–16 | Sobel_X, Sobel_Y, Sobel_Mag | zarr sobel 배열 |
