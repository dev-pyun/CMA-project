# Landsat 8 Cloud/Shadow/Snow Detection Training Pipeline

이 문서는 Landsat 8 위성 영상 기반 구름, 그림자, 눈 탐지를 위한 모델의 학습 데이터(패치) 생성 방법 및 학습 실행 방법, 그리고 사용자가 조작 가능한 주요 파라미터들에 대해 설명합니다.

상세 변경 이력 및 설계 결정은 [WALKTHROUGH.md](WALKTHROUGH.md)를 참고하세요.

---

## Directory Structure
```
src/
├── train.py              # 메인 학습 스크립트
├── predict.py            # 추론 스크립트
├── label_generation.py   # Pseudo-label 생성 (stage N → N+1)
├── make_landsat_data.py  # GeoTIFF → Zarr 변환 진입점
├── pipeline.sh           # 전체 파이프라인 일괄 실행 스크립트
├── dataset/
│   ├── patch_dataset.py  # PyTorch Dataset (Zarr patch 로딩)
│   ├── network_input.py  # Band 선택 + 분광 지수 계산
│   └── transforms.py     # Data augmentation
├── network/
│   ├── model.py          # UNet wrapper + 학습/검증 로직
│   └── unet.py           # UNet 구현
├── utils/
│   ├── split_scene.py    # GeoTIFF → Zarr 패치 변환 핵심 로직
│   ├── qa_pixel_mapping.py  # QA_PIXEL bitmask → 바이너리 라벨
│   ├── dir_paths.py      # 경로 상수 (여기서 소스 경로 관리)
│   ├── experiment.py     # 실험 관리 + 체크포인트
│   ├── MFB.py            # Median Frequency Balancing (클래스 가중치)
│   ├── metrics.py        # Accuracy / IoU 계산
│   ├── csv_logger.py     # 학습 메트릭 CSV 저장
│   └── join_predictions.py  # 예측 후처리
├── label_code/           # napari 기반 수동 라벨링 도구 (label_code/README.md 참고)
├── data/
│   ├── TRAIN_ZARR/       # 학습용 Zarr 패치 (256×256)
│   └── VALIDATION_ZARR/  # 검증용 Zarr 패치
└── exp_data/             # 실험 결과 (모델, 로그)
```

## Data Source
- **Weddell Sea 원본 데이터**: `/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/`
  - 구조: `{year}/{month}/{date}/{scene_id}/` (예: `2020/01/20200101/LC08_L1GT_.../`)
  - 데이터 복사 없이 직접 읽어 Zarr 패치 생성

## Label Scheme

### 학습 파이프라인 / Zarr patch 저장 형식 (QA_PIXEL 자동 생성)
| 값 | 의미 | loss |
|----|------|------|
| 0  | No-Cloud (Clear / Snow / Water) | 포함 |
| 1  | Cloud (Cloud + Shadow + Cirrus + Dilated) | 포함 |
| 255 | No-Data / ignore | **무시** (ignore_index=255) |

### 수동 라벨링 (napari, label_scene.py 출력)
| 값 | 의미 | napari 키 |
|----|------|----------|
| 0  | 미라벨 (patch 저장 시 255로 remap) | `0` |
| 1  | water | `1` |
| 2  | snow / ice | `2` |
| 3  | cloud shadow (**명확한 경우만**) | `3` |
| 4  | cloud (opaque + cirrus + dilated) | `4` |
| 255 | 센서 fill (자동 마킹) | — |

remap 규칙 (scene_to_patches.py): `{1,2}→0`, `{3,4}→1`, `{0,255}→255`

---

## 1. 학습 데이터(패치) 생성

모델 학습 전에 GeoTIFF 원본 영상들을 읽어 Zarr(`*.zarr`) 형태의 패치 데이터로 변환해야 합니다.

### 스크립트 실행 방법

```bash
# 학습용(Train) 데이터 패치 생성
python make_landsat_data.py --mode train

# 검증용(Validation) 데이터 패치 생성
python make_landsat_data.py --mode test

# 커스텀 경로 지정
python make_landsat_data.py --mode train --path /path/to/your/landsat/scenes
```

> **참고**: 원본 씬 폴더에는 `*_B1.TIF` ~ `*_B7.TIF` 및 `*_QA_PIXEL.TIF`가 필수입니다. `*_B9.TIF`(Cirrus)는 선택 사항이며, 없으면 0으로 채워집니다.

### Zarr Patch Format
```
spectral/    → B1–B7 + B9 (uint16, H×W×8)
rgb/         → OpenCV RGB 퍼센타일 정규화 (float32, H×W×3)
hsv/         → OpenCV HSV (float32, H×W×3)
sobel/       → Sobel X / Y / Magnitude (float32, H×W×3)
qa_label/    → 바이너리 라벨 0/1/255 (uint8, H×W)
```

---

## 2. 모델 학습 방법

### 전체 파이프라인 일괄 실행 (`pipeline.sh`)

```bash
./pipeline.sh <실험명(exp_name)> [입력모드(input_mode)] [사용할_GPU_ID]
```

**실행 예시:**
```bash
./pipeline.sh weddell_exp1 "swirndsi" "0 1"
./pipeline.sh exp2 "all_derived" "0"
```

> Stage 0 (QA 바이너리 라벨) → Stage 3 (가장 큰 네트워크)까지 pseudo-label 생성과 학습이 순차 진행됩니다.

### 수동 실행 예시
```bash
python train.py -e my_experiment -st 0 -ip swirndsi -gpu 0
python label_generation.py -e my_experiment -st 0
python train.py -e my_experiment -st 1 -ip swirndsi -gpu 0
```

---

## 3. Validation 수동 라벨링

학습된 모델의 정량 평가를 위해 napari GUI로 직접 라벨을 만드는 방법입니다.
자세한 내용은 [label_code/README.md](label_code/README.md)를 참고하세요.

### 빠른 시작

```bash
cd /home/pyuncb/src/label_code
conda activate cloud_label   # 또는 cloud

# Step 1. 씬 준비 (FCI / CFMask / bands 생성, ~5분)
python prepare_scene.py \
    --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --out_dir   prepared/

# Step 2. napari GUI 라벨링 (MobaXterm X11 포워딩 필요)
python label_scene.py \
    --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2

# Step 3. 256×256 patch 분할
python scene_to_patches.py \
    --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path   labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --out_root     patches/
```

**napari 단축키:**

| 키 | 동작 |
|----|------|
| `4` | cloud 칠하기 |
| `3` | shadow 칠하기 (명확한 경우만) |
| `2` | snow / ice 칠하기 |
| `1` | water 칠하기 |
| `0` | 미라벨로 지우기 |
| `P` | Polygon mode |
| `N` | Paint mode (브러시) |
| `E` | Erase |

**GUI 실행 환경 (Windows):**
MobaXterm으로 SSH 접속 시 X11 포워딩이 자동으로 활성화됩니다.
- MobaXterm 다운로드: https://mobaxterm.mobatek.net/download.html
- 접속 후 `echo $DISPLAY` 로 `:0` 또는 `localhost:10.0` 확인

---

## 4. 주요 조작 파라미터 (`train.py`)

### A. 입력 모드 (`-ip` / `--inp_mode`)

| 프리셋 | 채널 구성 |
|--------|---------|
| `swirndsi` (기본) | B2–B7 + NDSI (7ch) |
| `all_derived` | 스펙트럼 + RGB + HSV + Sobel (17ch) |
| `rgb_hsv` | RGB + HSV (6ch) |
| `swir_sobel` | B5–B6 + Sobel (5ch) |
| `cirrus_ndsi` | B2–B7 + B9 + NDSI (8ch) |
| `all_cirrus` | B1–B7 + B9 (8ch) |
| `rgb` | B2–B4 (3ch) |

### B. 학습 하이퍼파라미터

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `-lr` | 학습률 | `1e-6` |
| `-bs` | 배치 사이즈 | `32` |
| `-ep` | 최대 에폭 수 | `400` |
| `--no_dropout` | 드롭아웃 비활성화 | — |
| `--no_aug` | 데이터 증강 비활성화 | — |

### C. 네트워크 / 파이프라인

| 인자 | 설명 |
|------|------|
| `-st` | Self-training 스테이지 (0–3) |
| `--full` | 가장 큰 네트워크로 단일 지도학습 |

### D. 하드웨어
```bash
python train.py -e my_exp -st 0 -ip swirndsi -gpu 0 1   # 다중 GPU
```

---

## Training Pipeline

| Stage | 네트워크 | 라벨 소스 | 데이터 |
|-------|---------|-----------|-------|
| 0 | depth=5, filters=16 | QA_PIXEL (binary) | stage_0.txt |
| 1 | depth=5, filters=32 | pseudo-label | stage_0+1.txt |
| 2 | depth=6, filters=24 | pseudo-label | stage_0+1+2.txt |
| 3 | depth=6, filters=32 | pseudo-label | 전체 |

---

## 5. 결과 시각화

### 씬 전체 비교: Fmask vs 모델 예측 vs Ground Truth (`compare_scene.py`)

학습된 모델의 예측 결과를 씬 단위로 시각화합니다.
왼쪽: Fmask(QA_PIXEL), 가운데: 모델 예측, 오른쪽: 수동 라벨(Ground Truth)로 구성된 3-panel PNG를 저장합니다.

```bash
cd /home/pyuncb/src

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

python compare_scene.py \
    --scene_dir  $WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --exp        swirndsi_trial2_stage0 \
    --gpu        0 \
    --out        vis_output/
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--scene_dir` | 원본 Landsat 씬 디렉토리 | 필수 |
| `--label_path` | 수동 라벨 GeoTIFF 경로 | 필수 |
| `--exp` | 실험 이름 (`exp_data/` 하위) | 필수 |
| `--gpu` | GPU ID | `0` |
| `--out` | 결과 PNG 저장 디렉토리 | `vis_output/` |

출력: `vis_output/{scene_id}_comparison.png`
씬이 큰 경우(~7000×7000 px) 자동으로 다운샘플해서 저장합니다.

---

### 패치 단위 비교: Fmask vs 모델 예측 vs Ground Truth (`visualize_comparison.py`)

개별 val/test zarr 패치에 대해 3-panel 비교 이미지를 생성합니다.
원본 씬을 재스캔해 패치 좌표를 찾고 QA_PIXEL에서 Fmask를 읽어옵니다.

```bash
cd /home/pyuncb/src

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

# 단일 패치
python visualize_comparison.py \
    --patch data/VALIDATION_ZARR/LC08_L1GT_188114_20201114_20210315_02_T2_PATCH5.zarr \
    --exp   swirndsi_trial2_stage0 \
    --gpu   0

# VALIDATION_ZARR 디렉토리에서 6개 랜덤 샘플
python visualize_comparison.py \
    --patch  data/VALIDATION_ZARR/ \
    --exp    swirndsi_trial2_stage0 \
    --sample 6 \
    --gpu    0 \
    --out    vis_output/
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--patch` | zarr 패치 경로 또는 VALIDATION_ZARR 디렉토리 | 필수 |
| `--exp` | 실험 이름 (`exp_data/` 하위) | 필수 |
| `--label_dir` | 수동 라벨 TIF 디렉토리 | `label_code/labels` |
| `--scene_dir` | 씬 디렉토리 직접 지정 (생략 시 자동 탐색) | — |
| `--gpu` | GPU ID | `0` |
| `--out` | 결과 PNG 저장 디렉토리 | `vis_output/` |
| `--sample` | 디렉토리 지정 시 랜덤 샘플 수 | — |

출력: `vis_output/{scene_id}_PATCH{n}_comparison.png`

> **참고**: 패치마다 원본 씬을 재스캔하므로 패치당 10~30초 소요됩니다.
> 씬 전체 비교는 `compare_scene.py`를 사용하세요.

---

### 패치 단위 검사 (`inspect_zarr.py`)

개별 zarr 패치의 내용과 라벨 분포를 확인합니다.

```bash
# 단일 패치 내용 출력 (텍스트)
python inspect_zarr.py data/VALIDATION_ZARR/LC08_L1GT_188114_..._PATCH5.zarr

# 시각화 이미지 저장
python inspect_zarr.py data/VALIDATION_ZARR/LC08_L1GT_188114_..._PATCH5.zarr --save

# 여러 패치 랜덤 샘플링
python inspect_zarr.py data/VALIDATION_ZARR/ --sample 9 --save --out output_vis/
```

---

## GitHub 토큰 발급 및 push 설정

GitHub에 코드를 push할 때 HTTPS 인증이 필요합니다. 비밀번호 대신 Personal Access Token(PAT)을 사용합니다.

### 토큰 발급 방법

1. GitHub 로그인 → 우측 상단 프로필 → **Settings**
2. 좌측 메뉴 하단 → **Developer settings**
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
4. Note에 이름 입력 (예: `planck-server`)
5. Expiration 설정 (90일 권장)
6. Scope 체크:
   - `repo` (전체 체크)
7. **Generate token** 클릭 → 토큰 복사 (이 창 닫으면 다시 볼 수 없음!)

### 서버에 토큰 등록

```bash
# 방법 1: remote URL에 토큰 포함 (간단, 보안 주의)
git remote set-url origin https://<TOKEN>@github.com/dev-pyun/CMA-project.git

# 방법 2: credential helper로 캐싱 (권장)
git config --global credential.helper store
git push origin main
# → Username: dev-pyun
# → Password: <TOKEN> 입력
# 이후 자동 저장됨
```

### 토큰 만료 시

```bash
# 새 토큰 발급 후 다시 등록
git remote set-url origin https://<NEW_TOKEN>@github.com/dev-pyun/CMA-project.git
```
