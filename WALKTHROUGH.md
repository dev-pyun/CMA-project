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

---

## [2026-05-06] Train/Val/Test 데이터셋 구성 확정 + 학습 데이터 생성 준비

### 데이터셋 전체 씬 목록 (월별)

| 월 | Train | Val | Test |
|----|-------|-----|------|
| **1월** | `170110_20200101` (1) | `199110_20200128` (1) | `187116_20200124` (1) |
| | `169109_20200110` (3+1) | | |
| | `202114_20200117` (3+1) | | |
| | `183111_20200128` (4) | | |
| **2월** | `205111_20200207` (1) | `177110_20200219` (1) | `177110_20200203` (1) |
| | `168110_20200220` (1) | `171110_20200225` (1+2) | |
| | `184112_20200220` (3) | | |
| | `166109_20200222` (4) | | |
| **3월** | `179111_20200304` (1) | `165110_20200302` (3?) | `184109_20200307` (1) |
| | `198110_20200309` (1) | | `175109_20200308` (1+2) |
| | `212108_20200311` (5) | | |
| | `205098_20200326` (1+2) | | |
| **4월** | `209098_20200407` (2) | `181098_20200419` (1) | — |
| **10월** | `202114_20201015` (1) | `200111_20201017` (1+2) | `207105_20201018` (1) |
| | `200112_20201017` (1) | | |
| | `207112_20201018` (3+2) | | |
| | `221097_20201020` (4) | | |
| | `202113_20201031` (1) | | |
| | `218104_20201031` (4) | | |
| **11월** | `160110_20201110` (1) | `188114_20201114` (1) ✓ | — |
| | `188115_20201114` (1) | | |
| | `188116_20201114` (1) | | |
| | `160109_20201126` (5) | | |
| | `160110_20201126` (5) | | |
| | `188113_20201130` (3) | | |
| **12월** | `195115_20201201` (3) | `199105_20201213` (1) | `195110_20201201` (1) |
| | `209112_20201203` (5) | | `181114_20201215` (1) |
| | `189114_20201207` (4) | | |
| | `184111_20201220` (1) | | |
| **합계** | **29** | **8** | **7** |

카테고리: 1=과소탐지, 2=과대탐지, 3=그림자과소, 4=구름정탐지, 5=SKC정탐지

---

### 월별 씬 수 요약

| 월 | Train | Val | Test | 합계 |
|----|-------|-----|------|------|
| 1월 | 4 | 1 | 1 | **6** |
| 2월 | 4 | 2 | 1 | **7** |
| 3월 | 4 | 1 | 2 | **7** |
| 4월 | 1 | 1 | 0 | **2** |
| 10월 | 6 | 1 | 1 | **8** |
| 11월 | 6 | 1 | 0 | **7** |
| 12월 | 4 | 1 | 2 | **7** |
| **합계** | **29** | **8** | **7** | **44** |

---

### 카테고리 분포 (Train)

| 카테고리 | 씬 수 | 비율 |
|----------|-------|------|
| 1 / 1+2 혼재 | 13 | 45% |
| 2 (pure) | 1 | 3% |
| 3 / 3+1 혼재 | 6 | 21% |
| 4 | 5 | 17% |
| 5 | 4 | 14% |
| 오탐지 계열 합 | 20 | 69% |
| 정탐지 계열 합 | 9 | 31% |

Val·Test는 전 씬 수동 픽셀 라벨링 (QA_PIXEL 사용 안 함)

---

### 수정 파일

**`utils/split_scene.py`**:
- `import shutil` 추가
- `split_scene_to_patches()` — done marker + partial cleanup 로직 추가
  - `{out_folder}/{scene_name}.done` 존재 → skip (return -1)
  - `.done` 없고 `PATCH*.zarr` 존재 → 부분 처리된 것으로 판단, 전부 삭제 후 재처리
  - 완료 시 `.done` 파일에 패치 수 기록
- `make_patches()` — skip 씬 카운트 분리 출력
- `os.walk(followlinks=True)` — symlink 씬 디렉토리 탐색 가능하도록 수정
  - 버그: 기존 followlinks 없음 → TRAIN/ symlink 무시 → 씬 1개만 인식
  - 수정 후 29개 전부 인식

---

### 신규 파일

**`make_train_zarr.sh`**:
- `data/TRAIN/` 에 29개 씬 심볼릭 링크 생성 (이미 존재하면 skip)
- 소스 경로 존재 여부 검증
- `python -m utils.split_scene --mode train` 실행

**`label_code/VAL_TEST_LABELING.md`**:
- Val/Test 14개 씬 전체 라벨링 워크플로우 명령어 모음
- Step 1 (prepare_scene.py) → Step 2 (label_scene.py / X11 필요) → Step 3 (scene_to_patches.py)

---

### 현재 상태

- followlinks 수정 후 Train Zarr 재실행 시 나머지 28개 처리 (`160109`는 `.done`으로 skip)
- Val/Test 14개 씬 수동 라벨링 진행 예정 (`label_code/VAL_TEST_LABELING.md` 참조)

---

## [2026-05-06] label_scene.py — CFMask 초기화 기능 추가

### 배경

수동 라벨링 시 빈 캔버스에서 시작하면 모든 픽셀을 직접 칠해야 해서 시간이 과다 소요됨.
CFMask(QA_PIXEL 기반) 결과를 초기 라벨로 불러온 뒤, 오탐·미탐 영역만 수정하는 방식으로 개선.

### 수정 파일

**`label_code/label_scene.py`**:
- `_CFMASK_TO_LABEL` lookup table 추가
  - cfmask 1(cloud) → label 4(cloud) 만 초기화
  - shadow / snow / water / clear → label 0(미라벨) 유지
  - cfmask 255(fill) → label 255(fill)
- `cfmask_to_init_labels(cfmask)` 함수 추가 — lookup table 인덱싱으로 O(1) 리맵
- `launch_napari()` 에 `init_from_cfmask` 파라미터 추가
- CLI `--init_cfmask` 플래그 추가

**`label_code/VAL_TEST_LABELING.md`**:
- Step 2의 모든 `label_scene.py` 명령에 `--init_cfmask` 추가

### 사용 예

```bash
# cloud 영역이 미리 칠해진 상태로 시작, shadow·snow·water 직접 수정
python label_scene.py --prepared_dir prepared/LC08_L1GT_... --init_cfmask

# 중단 후 이어서 작업 (저장된 라벨 우선, init_cfmask 무시)
python label_scene.py --prepared_dir prepared/LC08_L1GT_... --resume
```

### 현재 상태

- Val/Test 14개 씬 라벨링 진행 중 (`--init_cfmask` 적용)

---

## [2026-05-07] 임계값 segmentation 테스트 + 패치 위치 오버뷰

### 신규 파일

**`label_code/test_segmentation.py`**:
- 밝기/NDSI/NDWI/Cirrus/BT 임계값 기반 cloud segmentation 계산
- napari에서 CFMask ref + Threshold Seg 레이어 동시 표시 → 토글 비교
- CLI 파라미터: `--bright`, `--cirrus`, `--ndsi`, `--ndwi`, `--bt`

```bash
conda activate napari_env
cd /home/pyuncb/src/label_code
python test_segmentation.py --prepared_dir prepared/LC08_L1GT_...
python test_segmentation.py --prepared_dir prepared/LC08_L1GT_... --bright 0.20 --cirrus 0.04
```

### 수정 파일

**`inspect_zarr.py`**:
- 패치 이름에서 `scene_id` + `patch_idx` 파싱 (`parse_patch_name()`)
- FCI 자동 탐색: `label_code/prepared/{scene_id}/fci.tif` (기본 경로, `--fci_dir`로 변경 가능)
- `compute_patch_bbox()`: patch_idx → 씬 내 (row_start, col_start) 좌표 계산
- `make_scene_overview()`: FCI 썸네일(512px 이하) 위에 빨간 박스로 패치 위치 표시
- 시각화 하단에 씬 오버뷰 행 자동 추가 (FCI 없으면 조용히 스킵)

```bash
# 씬 oFCI 없는 TRAIN 패치 → 오버뷰 없이 정상 출력
python inspect_zarr.py data/TRAIN_ZARR/..._PATCH5.zarr --save

# prepared FCI 있는 VAL 패치 → 씬 오버뷰 + 빨간 박스 자동 표시
python inspect_zarr.py data/VALIDATION_ZARR/..._PATCH5.zarr --save
```

---

## 2026-05-08 | scene_to_patches.py — 기존 패치 건너뛰기 / --overwrite 플래그 추가

### 배경
씬 처리 도중 중단되거나, 이미 생성된 씬을 재실행할 때 기존 패치를 덮어쓰지 않고 이어서 생성하는 기능 요청.

### 구현 내용

**`label_code/scene_to_patches.py`**:
- `process_scene()` 에 `overwrite: bool = False` 파라미터 추가
- 패치 저장 직전 `patch_path.exists() and not overwrite` 체크 → 존재 시 `n_saved`만 증가하고 실제 쓰기 건너뜀
  - 그리드 순서가 결정론적이므로 `n_saved` 인덱스 일관성 유지됨
- `n_skipped_existing` 카운터 추가, 완료 메시지에 출력
- argparse에 `--overwrite` 플래그 추가 (`action="store_true"`, 기본값 `False`)

### 사용법
```bash
# 기본 (이미 있는 패치 건너뜀 — 이어쓰기)
python scene_to_patches.py --scene_dir $WEDDELL/... --label_path labels/...tif --split val

# 기존 패치 덮어쓰기
python scene_to_patches.py --scene_dir $WEDDELL/... --label_path labels/...tif --split val --overwrite
```

---

## 2026-05-08 | MFB NUM_CLASSES 버그 수정

### 문제
`utils/MFB.py`의 `NUM_CLASSES = 6` 이 구버전(6-class) 코드 잔재로 남아 있어,
`get_MFB_weights()` 가 6-element 가중치 벡터를 반환했지만 모델은 2-class(binary)라서
`F.cross_entropy` 에서 `RuntimeError: weight tensor should be defined either for all or no classes` 발생.

### 수정
- `utils/MFB.py:18` — `NUM_CLASSES = 6` → `NUM_CLASSES = 2`

---

## 2026-05-08 | 결과 시각화 스크립트 추가

### compare_scene.py (신규)
씬 전체를 대상으로 **Fmask / 모델 예측 / Ground Truth** 3-panel PNG 생성.

- 씬 전체 밴드(B1–B7, B9)를 로드해 256×256 패치로 분할 후 모델 추론
- QA_PIXEL → `qa_pixel_to_binary()` 로 Fmask 생성
- 수동 라벨 GeoTIFF를 `{1,2}→0, {3,4}→1, {0,255}→255` 로 remap해 GT 생성
- 씬이 큰 경우(>2000px) 자동 다운샘플링

```bash
python compare_scene.py \
    --scene_dir  $WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --exp        swirndsi_trial2_stage0 \
    --gpu        0
```

### visualize_comparison.py (신규)
개별 val/test zarr 패치에 대한 3-panel 비교.
씬을 재스캔해 패치 좌표를 찾고 QA_PIXEL을 읽어 Fmask를 생성함.
패치당 씬 재스캔이 필요해 느리므로 씬 단위 비교는 `compare_scene.py` 권장.

```bash
# 단일 패치
python visualize_comparison.py \
    --patch data/VALIDATION_ZARR/LC08_L1GT_188114_20201114_20210315_02_T2_PATCH5.zarr \
    --exp   swirndsi_trial2_stage0 \
    --label_dir label_code/labels

# 디렉토리에서 랜덤 샘플 6개
python visualize_comparison.py \
    --patch data/VALIDATION_ZARR/ \
    --exp   swirndsi_trial2_stage0 \
    --label_dir label_code/labels \
    --sample 6
```

---

## 2026-05-08 | train.py — validation loader batch_size 수정

### 문제
`setup_data(mode='test', path=VALID_PATH)` 호출 시 `batch_size` 인자가 누락되어
validation DataLoader가 기본값(1)으로 동작하고 있었음.
학습 1178개 배치 / validation 1178개 배치로 보여 validation이 느렸던 원인.

### 수정
- `train.py:119` — `setup_data(mode='test', path=VALID_PATH)` →
  `setup_data(args.batch_size, mode='test', path=VALID_PATH)`

batch_size=32 적용 시 validation이 ~37배 빠르게 진행됨 (1178 배치 → ~37 배치).

---

## 2026-05-08 | TRAINING_GUIDE.md 신규 생성

### 내용
Self-training 파이프라인 전체를 설명하는 종합 문서.

- **파이프라인 개요**: 4-stage 진행 흐름 (Stage 0 QA_PIXEL → Stage 1–3 pseudo-label)
- **데이터 포맷**: Zarr 패치 구조, 17채널 입력 텐서 레이아웃
- **U-Net 아키텍처**: stage별 depth/filter 크기, 전체 forward 흐름 ASCII 다이어그램
- **Median Frequency Balancing (MFB)**: 클래스 불균형 대응 가중치 계산 방법
- **Early Stopping**: patience=10, val mIoU moving average 기준
- **Pseudo-label 생성**: confidence threshold 0.65, nodata 전파 로직
- **Data Augmentation**: 학습 시 적용되는 변환 종류
- **전체 실행 예시**: stage 0부터 stage 3까지 커맨드 순서

위치: `/home/pyuncb/src/TRAINING_GUIDE.md`

---

## 2026-05-08 | README.md — 결과 시각화 섹션 추가

### 추가 내용

README.md에 **"5. 결과 시각화"** 섹션 추가:
- `compare_scene.py`: 씬 전체 Fmask / 모델 예측 / GT 3-panel 비교 사용법
- `visualize_comparison.py`: 개별 zarr 패치 단위 3-panel 비교 사용법 (단일 / 랜덤 샘플링)
- `inspect_zarr.py`: 패치 내부 시각화 사용법


---

## 2026-05-11 | split_scene.py — fill 픽셀 skip 제거 (ignore 처리로 전환)

### 변경 이유
기존에는 QA_PIXEL fill 픽셀(label=255)이 하나라도 있는 패치를 전체 버렸음.
씬 경계 패치의 경우 95%+ 유효 픽셀을 가지고 있어도 전부 폐기되는 낭비 발생.
`cross_entropy(ignore_index=255)`가 이미 255 픽셀을 loss에서 제외하므로,
fill 픽셀을 그냥 255로 두고 저장해도 학습에 악영향 없음.

### 수정 (`utils/split_scene.py:282-286`)
```python
# 이전
if np.any(label == BINARY_NODATA):
    n_skipped += 1
    patch_pbar.update(1)
    continue

# 이후
# fill pixels remain as 255 (ignored in loss via ignore_index=255)
```

### 효과
- train 패치 수 증가 예상 (기존 5,110개 → 재생성 후 확인 필요)
- fill 픽셀은 label=255로 유지 → loss 계산에서 자동 제외

---

## 2026-05-11 | scene_to_patches.py — min_valid_frac 기본값 변경 (5% → 30%)

### 변경 이유
labeled 비율 5~10%짜리 패치는 한 패치(65536px)에서 ignore(255)가 90%+ 이상이라
val mIoU 계산에 기여하는 픽셀이 ~3000개 미만으로 통계적으로 노이즈에 가까움.
30% 기준으로도 1030개 패치가 유지되며, 고아 패치 문제를 피하려면 전체 삭제 후 재생성 필요.

### 수정
- `label_code/scene_to_patches.py`
  - `process_scene()` 기본값: `min_valid_frac=0.05` → `0.30`
  - argparse 기본값 동일하게 변경

### 주의
`--overwrite`는 덮어씌우는 방식이라, threshold 변경 후 재생성 시 기존 패치를 전부 삭제하고 실행해야 함:
```bash
rm -rf data/VALIDATION_ZARR/*.zarr
python label_code/scene_to_patches.py --scene_dir ... --label_path ... --split val
```

---

## 2026-05-11 | visualize_comparison.py — GT 오프셋 진단 + val 패치 stale 문제 발견

### 진단 과정
사용자가 GT가 Fmask 대비 한 칸 오른쪽에 있는 것처럼 보인다고 보고.
`diagnose_label_offset.py` 스크립트로 진단한 결과:
- label TIF와 raw band의 좌표계 완전 일치 (픽셀 오프셋 0,0) → 오프셋 버그 아님
- PATCH5는 all-cloud 패치라 진단 불가 (any cloud-heavy candidate → score 1.0)
- 전체 val 패치 no_cloud 집계 결과 `0` → 원인 추적

### 원인
**zarr 패치 생성 이후에 no-cloud 영역(water/snow)을 추가 라벨링**하여
stale zarr(옛 label 기준)와 현재 label TIF가 불일치.

타임스탬프 비교:
- `171110` STALE: zarr 13:06 < label 13:27
- `177110` STALE: zarr 12:33 < label 13:47
- `188114` STALE: zarr 2026-05-07 < label 2026-05-11
- `181098` 패치 없음

### diagnose_label_offset.py (신규)
특정 zarr 패치의 label이 올바른 지리 위치에서 읽혔는지 판정하는 진단 스크립트.
- label TIF vs raw band 공간 정보 비교
- (row,col) / (row,col±stride) / (row±stride,col) 위치 후보별 일치율 계산
- Fmask와의 일치율도 함께 출력

---

## 2026-05-11 | val 패치 재생성 — stale label 문제 수정

### 문제
zarr 패치 생성 후 no-cloud 영역(water/snow)을 추가 라벨링했으나, 기존 패치에는 반영되지 않음.
`scene_to_patches.py` 실행 시점보다 나중에 label TIF가 수정된 씬(stale) 확인.

| 씬 | 판정 | zarr 생성일 | label 수정일 |
|---|---|---|---|
| `165110` | OK | 2026-05-07 17:56 | 2026-05-07 16:50 |
| `171110` | **STALE** | 2026-05-08 13:06 | 2026-05-08 13:27 |
| `177110` | **STALE** | 2026-05-08 12:33 | 2026-05-08 13:47 |
| `188114` | **STALE** | 2026-05-07 17:47 | 2026-05-11 10:44 |
| `181098` | **패치 없음** | — | — |
| `199110` | OK | 2026-05-07 17:49 | 2026-05-07 17:45 |

### 조치
stale 3개 씬 `--overwrite` 재생성 + `181098` 신규 생성

```bash
python label_code/scene_to_patches.py --scene_dir $WEDDELL/.../171110... --label_path ... --split val --overwrite
python label_code/scene_to_patches.py --scene_dir $WEDDELL/.../177110... --label_path ... --split val --overwrite
python label_code/scene_to_patches.py --scene_dir $WEDDELL/.../188114... --label_path ... --split val --overwrite
python label_code/scene_to_patches.py --scene_dir $WEDDELL/.../181098... --label_path ... --split val
```

### 결과
- 전체 val 패치: 1178 → **1519개** (181098 씬 341개 신규)
- `181098` no_cloud=0 — water/snow 라벨 미입력 상태, 추후 재라벨링 필요

**주의**: 현재 진행 중인 stage 1 학습은 변경 전 val 데이터로 진행 중. 다음 학습부터 반영됨.

---

## 2026-05-11 | compare_stages.sh 신규 + visualize_comparison.py 파일명 변경

### compare_stages.sh (신규)
랜덤 샘플링한 패치에 대해 stage0~3 전부를 한 번에 비교 시각화하는 파이프라인.

- 패치를 **한 번만** 샘플링한 뒤 stage0→3 순서로 `visualize_comparison.py` 호출
- 실험 디렉토리(`exp_data/{exp}_stageN`)가 없는 stage는 자동 스킵
- 출력: `vis_output/{exp_base}_stages_{YYYYMMDD_HHMMSS}/`

```bash
./compare_stages.sh swirndsi_trial2          # 기본 5개 샘플, GPU 0
./compare_stages.sh swirndsi_trial2 8 0      # 8개 샘플
./compare_stages.sh swirndsi_trial2 5 "0 1"  # 멀티 GPU
```

### visualize_comparison.py 파일명 변경
- 기존: `{scene_id}_PATCH{patch_idx}_comparison.png`
- 변경: `{scene_id}_PATCH{patch_idx}_{exp_name}_comparison.png`

stage가 다른 결과물이 같은 폴더에 저장될 때 덮어쓰기 방지 및 stage 식별 용이.

---

## 2026-05-11 | visualize_comparison.py — 씬 오버뷰 패널 추가 (2×3 그리드)

### 변경 내용

- 레이아웃을 **1×3 → 2×3** 으로 변경 (figsize 18×6 → 18×12)
- **하단 (1,1) 위치**에 씬 전체 B4 밴드 썸네일 + 패치 위치 빨간 박스 패널 추가
  - (1,0), (1,2) 는 비워둠 (`axis('off')`)
- `make_scene_overview(scene_dir, row, col)` 함수 추가
  - B4 밴드를 rasterio로 로드, 최대 800px 썸네일로 다운샘플 (rasterio `out_shape` 이용)
  - 패치 위치(row, col)를 스케일 변환 후 빨간 사각형 테두리 그리기
  - B4 파일 없을 시 `None` 반환 → 패널 빈칸으로 처리
- `utils.split_scene.find_band_file` import 추가

---

## 2026-05-11 | visualize_comparison.py — MIN_VALID_FRAC 불일치 버그 수정

### 문제
씬 오버뷰의 패치 위치(빨간 박스)가 잘못된 위치에 표시됨.

### 원인
`find_patch_coords()`가 패치 인덱스(PATCH121 등)로부터 (row, col) 좌표를 역산할 때
`MIN_VALID_FRAC = 0.05` (5%) 를 사용하고 있었으나,
`scene_to_patches.py`의 기준을 **0.30** (30%)으로 올린 이후 불일치 발생.

유효 라벨 비율 5~30% 사이의 패치들이 `scene_to_patches.py`에서는 건너뛰어지지만
`find_patch_coords()`에서는 포함되어 카운트 → 패치 인덱스 어긋남 → 좌표 오류.

### 수정 (`visualize_comparison.py`)
- `MIN_VALID_FRAC = 0.05` → `0.30`


---

## 2026-05-13 | visualize_comparison.py — 씬 오버뷰 FCI 변경 + fill 마스크 수정

### 변경 내용

**씬 오버뷰 FCI 변환** (`make_scene_overview()`):
- 기존: B4 단일 밴드 흑백 썸네일
- 변경: FCI (False Color Infrared) 컬러 합성
  - R = B5 (NIR), G = B4 (Red), B = B3 (Green)
  - 구름/눈/얼음이 붉은색, 식생이 초록색으로 보여 구름 구별이 용이
- 패널 제목도 "Scene Overview" → "Scene FCI Overview"로 변경

**fill 픽셀 마스크** (`visualize_patch()`):
- 모델 예측에서 fill 픽셀(swath 밖, 스펙트럴=0)이 no-cloud(0)로 예측되는 문제 수정
- `fill_mask = (fmask == 255) | (gt == 255)` → `pred[fill_mask] = 255`로 덮어씌움
- 시각화 전용 수정, 모델 자체에는 영향 없음


---

## 2026-05-13 | compare_scene.py — 실행 방법 주석 보강

docstring에 씬별 경로, 인자 설명, 6개 val 씬 목록, 출력 형식을 상세히 기술.

---

## 2026-05-13 | Gini 기반 패치 샘플링

### 배경
랜덤 샘플링 시 cloud/no-cloud 비율이 극단적인 패치(거의 전부 cloud이거나 전부 no-cloud)가
선택되는 경우 비교 시각화의 의미가 떨어짐.

### Gini impurity 정의
3-class (0=no-cloud, 1=cloud, 255=ignore) Gini impurity:

    gini = 1 - Σp_i²   (i ∈ {0, 1, 255})

- 범위: [0, 2/3 ≈ 0.667]
- 0 = 단일 클래스 (모두 같은 레이블)
- 0.667 = 세 클래스가 1/3씩 균등 혼합

### 수정 파일

**`visualize_comparison.py`**:
- `compute_gini(patch_path)` 함수 추가 — zarr label 읽어 3-class Gini 계산
- `sample_by_gini(patch_dirs, n, min_gini)` 함수 추가 — 필터 후 랜덤 샘플링
- `--min_gini` 인자 추가 (기본 0.0=필터 없음, 권장 0.1~0.3)
- `--list_only` 플래그 추가 — 경로만 출력하고 추론 실행 안 함 (bash 파이프용)

**`compare_stages.sh`**:
- `[min_gini]` 5번째 인자 추가 (기본 0.1)
- `shuf -n N` → Python `--list_only` 호출로 교체
  - `visualize_comparison.py --list_only`가 Gini 필터+샘플링 결과를 stdout으로 출력
  - `mapfile`로 받아서 PATCHES 배열에 저장


---

## 2026-05-13 | compare_scene.py — GT 선택적, 씬별 폴더 분리 저장

### 변경 내용

- `--label_path` 인자를 **선택 사항**으로 변경 (train 씬 지원)
- 단일 합성 이미지 → **씬별 폴더 + 개별 파일** 구조로 변경
  ```
  vis_output/{scene_id}/fmask.png
  vis_output/{scene_id}/model_{exp_name}.png
  vis_output/{scene_id}/ground_truth.png  ← label_path 있을 때만
  ```
- `_save_panel()` 헬퍼 함수 추가 (단일 패널 저장)
- `compare_scene()` 시그니처: `label_path` 위치를 키워드 인자로 변경


---

## 2026-05-13 | compare_scene.py — 씬 가장자리 미처리 strip 수정

### 문제
추론 루프가 `range(0, H-256+1, 256)` 으로 정의되어 마지막 stride 이후 edge 영역이 미처리됨.
188115 씬 기준: 우측 105px + 하단 251px 가 pred_map 초기값(255=회색)으로 남음.
Weddell Sea ocean 위에 dark grey(0.10) 오버레이가 55% alpha로 얹히면 배경(어두운 물)이 그대로 비쳐보여 "투명하게 뚫린" 것처럼 보임.

### 수정
`make_coords()` 함수 추가: 마지막 좌표가 씬 끝까지 커버 못 할 경우 `length - PATCH_SIZE` 를 끝에 추가.



---

## 2026-05-13 | compare_scene.py — cloud 색상 + fill masking 수정

### 문제 (연회색 현상)
cloud 오버레이 색상이 순백색 (1.0, 1.0, 1.0) 이라, alpha blending 시 어두운 배경(water, cloud shadow)에서는 연회색으로 나타남.
- 밝은 얼음 배경(0.9) + white × α=0.8 → 0.98 = 흰색 ✓
- 어두운 물 배경(0.05) + white × α=0.8 → 0.81 = **연회색** ✗

또한 fill 픽셀(QA_PIXEL bit 0, spectral=0)이 모델을 통과해 class 0 또는 1로 예측되어, Fmask 패널의 진회색(255) 표시와 불일치.

### 수정
1. **cloud 색상 변경**: `(1.0, 1.0, 1.0)` 흰색 → `(0.53, 0.81, 0.98)` 하늘색
   - 하늘색(saturated)은 어두운/밝은 배경 모두에서 동일하게 하늘색으로 보임
2. **fill pixel masking 추가**: 추론 직후 `pred[fmask == 255] = 255` 적용
   - fill 영역은 모델 예측 무시, no-data(진회색)로 표시

---

## [2026-05-13] 패치 경계 격자 아티팩트 제거 + TOA Reflectance 변환

### 배경 및 목표
- 모델 추론 시 256×256 패치 경계에서 격자(grid) 아티팩트 발생 확인
- 원인: stride=256으로 non-overlapping 패치 추론 시, 각 패치 경계 픽셀이 인접 패치의 실제 픽셀을 참조하지 못하고 zero padding으로 대체되어 경계 불연속성 발생
- 추가: DN값 대신 TOA Reflectance를 학습에 사용 (물리적 의미 + 씬 간 일관성 향상)

### 수정 내용

#### 1. `utils/split_scene.py`
- **258×258 패치 추출**: 256×256 중심 + 1px 실제 인접 픽셀 (씬 경계는 zero-fill)
  - `row_pad_start/end`, `col_pad_start/end`로 확장된 rasterio window 설정
  - `off_r/off_c`: 씬 경계 여부에 따라 canvas 내 쓰기 시작 위치 결정
  - `off = 1` when at scene edge (no border → canvas[0] stays 0), `off = 0` otherwise
- **TOA Reflectance 변환** 추가:
  - `_find_mtl_file(scene_dir)`: 씬 폴더 내 MTL JSON 탐색
  - `_load_sun_sin(scene_dir)`: MTL에서 태양 고도각 읽어 sin 값 반환 (없으면 1.0)
  - `_dn_to_toa_uint16(spectral, sun_sin)`: `clip((2e-5×DN − 0.1) / sin(θ), 0, 1) × 10000` → uint16 저장
  - 씬 단위로 `sun_sin` 계산 후 패치 루프에서 재사용
- Zarr 포맷 변경: spectral `(256,256,8) DN uint16` → `(258,258,8) TOA×10000 uint16`
- label은 `(256,256)` 그대로 유지 (stride 불변)

#### 2. `dataset/patch_dataset.py`
- 기존 zero padding 제거: `np.pad(full_input, ...)` 삭제
- label을 transforms **이전**에 258×258로 패딩 (NODATA=255)
  - transforms가 full_input(258×258)와 label(256→258)을 동일 크기로 처리 가능
- `/10000` 로딩 그대로 유지 → TOA reflectance [0,1] 자동 획득

#### 3. `compare_scene.py`
- `_dn_to_toa_uint16`, `_load_sun_sin` import 추가
- 씬 밴드 로드 직후 TOA 변환 적용 (sun_sin 포함)
- 패치 루프에서 zero padding → **실제 인접 픽셀 258×258 추출**:
  ```python
  ri0, ri1 = max(0, i-1), min(H, i+PATCH_SIZE+1)
  ci0, ci1 = max(0, j-1), min(W, j+PATCH_SIZE+1)
  patch_pad = np.zeros((PATCH_SIZE+2, PATCH_SIZE+2, ...), dtype=spectral.dtype)
  patch_pad[off_r:..., off_c:...] = spectral[ri0:ri1, ci0:ci1]
  ```

#### 4. `label_code/scene_to_patches.py`
- 258×258 real border 추출 로직 추가 (split_scene.py와 동일)
- `_dn_to_toa_uint16`, `_load_sun_sin` import 및 적용
- `from rasterio import windows as rwin` import 위치 함수 외부로 이동

### 기타 작업
- TRAIN 씬 폴더에 MTL JSON 이미 symlink로 존재 확인 (source가 Weddell Sea 원본)
- 기존 TRAIN_ZARR / VALIDATION_ZARR 전체 삭제 후 nohup으로 재생성 시작:
  ```bash
  nohup conda run -n remote python make_landsat_data.py --mode train > logs/make_train_zarr.log 2>&1 &
  ```
- VALIDATION 패치는 `label_code/scene_to_patches.py`로 별도 생성 (수동 라벨 기반)

### TOA Reflectance ×10000 uint16 저장 이유
- float32 저장 시 파일 크기 2배 → uint16 유지로 기존 크기 동일
- `(reflectance × 10000).uint16` 저장 후 `/10000` 로드 = reflectance [0,1]
- patch_dataset.py의 `/ 10000.0` 코드 변경 없이 자동 호환

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
