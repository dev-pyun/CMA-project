# WALKTHROUGH — Landsat 8 Cloud Detection Pipeline

---

## [2026-04-30] 패치 포맷 전환 (HDF5 → Zarr) + Binary 라벨 + 파생 피처 추가

### 배경 및 목표
- QA_PIXEL 6-class 라벨을 **binary cloud / no-cloud**로 단순화
- 패치 저장 시 학습에 활용 가능한 모든 파생 피처(RGB, HSV, Sobel)를 미리 계산해 저장
- 저장 포맷을 HDF5 → **Zarr + blosc/zstd 압축**으로 교체

### Zarr 포맷 도입의 주요 이점
1. **PyTorch DataLoader의 병렬 처리 성능 극대화 (GIL 병목 해소)**
   - HDF5(`h5py`)는 단일 파일 기반으로 I/O 시 Global Interpreter Lock (GIL)이 걸리거나 C 레벨의 lock이 발생하여 여러 개의 DataLoader worker가 동시에 읽을 때 심각한 병목이 발생합니다.
   - Zarr는 청크(chunk) 단위로 독립된 파일로 저장되므로, 여러 worker가 **동시에 lock 없이(lock-free) 데이터를 병렬로 읽을 수 있어 학습 속도가 크게 향상**됩니다.
2. **손상 및 안정성 개선**
   - HDF5는 파일 하나에 모든 데이터를 담고 있어 학습/패치 생성 중 프로세스가 강제 종료되면 전체 파일 구조가 손상(corrupted)될 위험이 큽니다.
   - Zarr는 메타데이터(json)와 청크 단위 파일 리스트 형태이므로, 쓰기 중단 시에도 파일 전체가 깨지는 현상을 방지할 수 있습니다.
3. **향상된 압축 속도 및 최신 코덱 지원**
   - HDF5의 기본 zlib/gzip 압축보다 훨씬 빠르고 최적화된 **Blosc + Zstandard(zstd) + bitshuffle 코덱**을 네이티브로 지원하여, 디스크 공간 절약은 물론 압축 해제 속도가 매우 빨라 랜덤 액세스 성능이 크게 개선됩니다.

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

#### 10. `inspect_zarr.py` (구 `inspect_h5.py`)
- zarr 패치 지원으로 전면 재작성
- binary 라벨 시각화 (No-Cloud/Cloud/No-Data 3색)
- spectral + RGB + HSV + Sobel Magnitude + NDSI 패널 표시
- `.zgroup` / `zarr.json` 파일 유무로 단일 패치 / 상위 디렉토리 자동 구분

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
python inspect_zarr.py data/TRAIN_ZARR/LC08_..._PATCH0.zarr
python inspect_zarr.py data/TRAIN_ZARR/ --sample 6 --save
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

---

## [2026-05-04] label_code 라벨 scheme 변경 — 6-class 수동 라벨 + binary remap

### 변경 이유
- napari 초기값이 0이므로, 0을 no-cloud로 쓰면 라벨링 전 전체 씬이 이미 labeled된 것처럼 보이는 혼동 발생
- 수동 라벨을 **6-class**로 세분화해 에러 분석 및 클래스별 성능 진단 가능하게 함
- Shadow는 cloud와 스펙트럼이 정반대(dark vs bright)이므로 분리 라벨링이 학습에 유리
- Cirrus는 시각적 구분이 어렵고 cloud와 같은 목적(masking)이므로 cloud(5)에 통합
- 학습 파이프라인은 그대로 binary (0=no-cloud, 1=cloud) 유지, remap은 patch 생성 시 처리

### 라벨 scheme

**napari 라벨링 (label_scene.py 출력):**
```
0   = 미라벨 (napari 기본값) → patch 저장 시 255(ignore)로 remap
1   = clear land
2   = water
3   = snow / ice
4   = cloud shadow  (명확한 경우만; 애매하면 0으로 두기)
5   = cloud (opaque + thin cirrus + dilated 포함)
255 = 센서 fill (자동 마킹)
```

**remap 규칙 (scene_to_patches.py):**
```
{1, 2, 3} → 0  (no-cloud)
{4, 5}    → 1  (cloud)
{0, 255}  → 255 (ignore)
```

### Shadow 라벨링 원칙
- CFMask overlay + FCI 영상이 **모두 어두운 경우만** 4(shadow) 라벨링
- Dark water / dark rock과 구분 불가한 픽셀 → 0(미라벨)으로 두면 ignore 처리
- Shadow 경계 픽셀도 0으로 두어도 됨 — 모델은 라벨된 픽셀만 학습

### 수정 파일

**`label_code/label_scene.py`**:
- `LABEL_CLASSES`: 6-class `{0:nodata, 1:clear, 2:water, 3:snow, 4:shadow, 5:cloud, 255:fill}`
- 레이어 이름: `"MY_LABELS (5=cloud 4=shadow 3=snow 2=water 1=clear 0=미라벨 255=fill)"`
- 단축키 안내: `5`=cloud, `4`=shadow, `3`=snow, `2`=water, `1`=clear, `0`=미라벨
- `launch_napari()`: `init_labels` 파라미터로 resume 모드 개선
- `save_labels()`: 0~5/255 그대로 저장 (remap은 scene_to_patches.py에서)

**`label_code/scene_to_patches.py`**:
- `LABEL_REMAP` / `VALID_LABEL_VALUES` 상수 추가
- `remap_labels()` 함수: `{1,2,3}→0`, `{4,5}→1`, `{0,255}→255`
- `valid_mask`: `np.isin(lr, {1,2,3,4,5})` — 미라벨(0)·fill(255) 제외
- `attrs`: `has_shadow`, `has_snow`, `has_water`, `has_clear`, `shadow/snow/water/clear_frac` 추가
- 통계 출력: 6-class 분류 기준

**`label_code/README.md`**:
- 클래스 표 2개 분리 (napari scheme / patch 저장 scheme)
- Shadow 라벨링 원칙 명시 (애매한 케이스 처리 방법 포함)
- 전체 워크플로우 업데이트

**`README.md`**:
- Label Scheme 섹션: 학습 파이프라인 scheme + 수동 라벨링 scheme + remap 규칙
- "Validation 수동 라벨링" 섹션 napari 단축키 업데이트
- "GitHub 토큰 발급 및 push 설정" 섹션 추가

---

## [2026-05-04] label_code clear land 클래스 제거

### 변경 이유
Weddell Sea 극지방 씬에서 clear land(노출 암석)가 거의 등장하지 않아 실질적으로 사용되지 않음. 불필요한 클래스 제거.

### 최종 scheme

**napari 라벨링:**
```
0 = 미라벨, 1 = water, 2 = snow/ice, 3 = shadow, 4 = cloud, 255 = fill
```

**remap (scene_to_patches.py):**
```
{1, 2} → 0 (no-cloud)
{3, 4} → 1 (cloud)
{0, 255} → 255 (ignore)
```

### 수정 파일
- `label_code/label_scene.py`: LABEL_CLASSES, 레이어 이름, 단축키 안내
- `label_code/scene_to_patches.py`: LABEL_REMAP, VALID_LABEL_VALUES, attrs, 통계 출력, docstring
- `label_code/README.md`: 클래스 표, 단축키 표, 통계 예시
- `README.md`: 수동 라벨링 scheme 표, 단축키 표

---

## [2026-05-04] scene_to_patches.py — Zarr 포맷으로 전환 + val/test 직접 저장

### 변경 이유
- 기존: HDF5(`/input`, `/label`)로 `label_code/patches/`에 저장 → 학습 파이프라인과 포맷 불일치
- Nambiar 논문에서 validation은 human-labeled 데이터를 매 epoch 사용
- 수동 라벨 패치가 `data/VALIDATION_ZARR/` / `data/TEST_ZARR/`에 학습 패치와 동일 포맷으로 저장되어야 함

### 밴드 구성 불일치 해결
- `prepared/bands.tif` (B2-B7+B9+B10, float32 TOA) ≠ 학습 패치 (B1-B7+B9, uint16 DN)
- 해결: `--scene_dir`로 원본 TIF에서 직접 읽도록 변경 → `split_scene.py` 함수 재사용

### 수정 파일

**`label_code/scene_to_patches.py`**:
- 완전 재작성: HDF5 → Zarr, `prepared_dir` → `--scene_dir` (원본 TIF)
- `sys.path`로 `utils/split_scene.py` import: `find_band_file`, `compute_rgb/hsv/sobel`, `save_patch_zarr`
- `--split val/test` 인자로 출력 경로 자동 결정 (`VALIDATION_ZARR` / `TEST_ZARR`)
- 유효 라벨 < 5% 패치는 모두 버림 (val/test는 train_aux 개념 없음)
- 출력 포맷: `spectral/rgb/hsv/sobel/label` (학습 패치와 100% 동일)

**`utils/dir_paths.py`**:
- `TEST_ZARR_PATH = data/TEST_ZARR` 추가
- `makedirs` 목록에 `TEST_ZARR_PATH` 추가

**`label_code/README.md`**:
- Step 3 완전 재작성: 새 사용법 + Zarr 포맷 설명 + 씬 수 가이드라인

---

## [2026-05-04] zarr v3 compressor 오류 수정

### 오류
`scene_to_patches.py` 실행 시 `save_patch_zarr`에서 실패:
```
TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'> instead.
ZarrUserWarning: The `compressor` argument is deprecated. Use `compressors` instead.
```
`remote` 환경의 zarr 3.1.6은 numcodecs.Blosc를 더 이상 압축기로 받지 않음.

### 수정 (`utils/split_scene.py`)
- `from numcodecs import Blosc` → `from zarr.codecs import BloscCodec`
- `_COMP_*` 정의를 `BloscCodec(cname=..., clevel=..., shuffle=...)` 로 교체
  - shuffle 값: `'bitshuffle'` / `'shuffle'` (문자열)
- `create_dataset` 호출의 `compressor=x` → `compressors=[x]` (리스트, zarr v3 API)

### 검증
`conda run -n remote python` 스모크 테스트 통과: save 후 open_group으로 shape 확인 OK

---

## [2026-05-04] HDF5 → Zarr 전환 후 잔존 코드 수정

### 수정 파일

**`network/model.py`**:
- `from numcodecs import Blosc` → `from zarr.codecs import BloscCodec`
- `Blosc(cname=..., shuffle=Blosc.BITSHUFFLE)` → `BloscCodec(cname=..., shuffle='bitshuffle')`
- `create_dataset` → `create_array`, `compressor=x` → `compressors=[x]`
- `data=` 사용 시 `shape=`, `dtype=` 동시 사용 불가 → 제거 (zarr v3 규칙)

**`utils/join_predictions.py`**:
- 전면 교체: h5py + `.h5` → zarr + `.zarr`
- `join_patches(zarr_dir, output_path, key)` 형태로 재작성
- `--h5_dir` → `--zarr_dir`, `--channel` → `--key` (`raw_prediction`, `pseudo_label` 등)

**`utils/split_scene.py`**:
- `create_dataset` → `create_array` (zarr v3에서 create_dataset deprecated)
- `create_array(data=x, chunks=..., compressors=[...])` — dtype/shape은 data에서 자동 추론

### zarr v3 create_array 규칙 요약
- `data=` 사용 시: `shape=`, `dtype=` 동시 사용 불가 — numpy 배열에서 자동 추론
- `compressor=` → `compressors=[codec]` (리스트)

### 검증
warnings-as-errors 모드 스모크 테스트 통과: spectral/label dtype·shape OK, pseudo_label write/read OK

---

## [2026-05-04] 세션 요약 — 수동 라벨링 파이프라인 완성 + zarr v3 호환성 확보

### 작업 개요

이번 세션에서 수동 라벨링 → validation/test 패치 생성 파이프라인 전체를 완성하고,
zarr v3 환경(`remote` conda env, zarr 3.1.6)과의 호환성을 확보했다.

---

### 1. label_code 라벨 scheme 최종 확정

**변경 흐름:**
- 기존 binary (0=no-cloud, 1=cloud) → napari 초기값 혼동 문제
- 6-class 수동 라벨 시도 (0=미라벨, 1=clear, 2=water, 3=snow, 4=shadow, 5=cloud)
- clear land 제거 (Weddell Sea 극지방에서 거의 미등장)

**최종 scheme:**

| napari 값 | 의미 | patch 저장 값 |
|-----------|------|--------------|
| 0 | 미라벨 (napari 기본값) | 255 (ignore) |
| 1 | water | 0 (no-cloud) |
| 2 | snow / ice | 0 (no-cloud) |
| 3 | cloud shadow (명확한 경우만) | 1 (cloud) |
| 4 | cloud (opaque + cirrus + dilated) | 1 (cloud) |
| 255 | 센서 fill (자동) | 255 (ignore) |

**Shadow 라벨링 원칙:** CFMask overlay + FCI 영상이 **모두 어두운 경우만** 라벨링.
Dark water/rock과 구분 불가한 픽셀은 0(미라벨)으로 두면 ignore 처리.

---

### 2. scene_to_patches.py 재작성

- **입력 변경**: `--prepared_dir` → `--scene_dir` (원본 Landsat TIF 직접 읽기)
  - 이유: `prepared/bands.tif`는 B2-B7+B9+B10 float32 TOA → 학습 패치(B1-B7+B9 uint16 DN)와 포맷 불일치
- **출력 변경**: `label_code/patches/` HDF5 → `data/VALIDATION_ZARR/` / `data/TEST_ZARR/` Zarr
  - 이유: Nambiar 논문에서 human-labeled validation을 매 epoch 사용 → 학습 패치와 동일 포맷 필요
- **필터링**: fill > 50% 스킵, 유효 라벨 < 5% 스킵
- **`utils/dir_paths.py`**: `TEST_ZARR_PATH` 추가

---

### 3. zarr v3 API 오류 3단계 수정

| 오류 | 원인 | 수정 |
|------|------|------|
| `TypeError: Expected a BytesBytesCodec` | zarr v3는 `numcodecs.Blosc` 미지원 | `BloscCodec(cname, clevel, shuffle)` |
| `ZarrDeprecationWarning: use compressors` | `compressor=` deprecated | `compressors=[codec]` (리스트) |
| `ZarrDeprecationWarning: use create_array` | `create_dataset` deprecated | `create_array(...)` |
| `ValueError: data + shape 동시 불가` | zarr v3 `create_array` 규칙 | `data=` 사용 시 `shape=`, `dtype=` 제거 |

**수정 파일**: `utils/split_scene.py`, `network/model.py`

---

### 4. 전체 zarr 잔존 코드 정리

- **`network/model.py`**: `generate_train_data()` 내 compressor 교체, `create_array` 변환
- **`utils/join_predictions.py`**: h5py + `.h5` 전면 제거 → zarr + `.zarr` 재작성
  - `--h5_dir` → `--zarr_dir`, `--channel` → `--key` (예: `raw_prediction`, `pseudo_label`)
- **`inspect_zarr.py`** (구 `inspect_h5.py`): `.zgroup` → `.zgroup` or `zarr.json` 체크 추가 (zarr v3 메타데이터 파일명 변경)

---

### 5. 첫 번째 validation 패치 생성 완료

씬: `LC08_L1GT_188114_20201114_20210315_02_T2` (6701×6811 px)

```
202 패치 저장
 57 스킵 (fill 비율 > 50%)
417 스킵 (유효 라벨 < 5%)
```

**spectral 배열 채널 구성 확인:**

| ch | 밴드 | min | max | nonzero |
|----|------|-----|-----|---------|
| 0 | B1 | 19718 | 24770 | 65536 |
| 1 | B2 | 19213 | 24824 | 65536 |
| 2 | B3 | 17741 | 23739 | 65536 |
| 3 | B4 | 17549 | 24398 | 65536 |
| 4 | B5 | 16485 | 24718 | 65536 |
| 5 | B6 | 8346 | 18862 | 65536 |
| 6 | B7 | 8237 | 18101 | 65536 |
| 7 | B9 | 5566 | 8573 | 65536 |

모든 채널 nonzero=65536 (256×256) — zero-fill 없이 전 픽셀 유효.

---

### 현재 상태 및 다음 단계

**완료:**
- [x] 수동 라벨링 scheme 확정 (5-class → binary remap)
- [x] scene_to_patches.py zarr 포맷 + val/test 경로 출력
- [x] zarr v3 API 완전 호환
- [x] 첫 번째 validation 씬 패치 생성 (202개)
- [x] inspect_zarr.py로 패치 내용 확인 방법 정립

**다음 단계:**
- [ ] validation/test 씬 추가 라벨링 (목표: 각 5–8 씬)
- [ ] 학습 데이터(TRAIN_ZARR) 생성 후 train.py stage 0 실행
