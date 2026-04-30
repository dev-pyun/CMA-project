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
├── label_code/           # napari 기반 수동 라벨링 도구
├── data/
│   ├── TRAIN_ZARR/       # 학습용 Zarr 패치 (256×256)
│   └── VALIDATION_ZARR/  # 검증용 Zarr 패치
└── exp_data/             # 실험 결과 (모델, 로그)
```

## Data Source
- **Weddell Sea 원본 데이터**: `/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/`
  - 구조: `{year}/{month}/{date}/{scene_id}/` (예: `2020/01/20200101/LC08_L1GT_.../`)
  - 데이터 복사 없이 직접 읽어 Zarr 패치 생성

## Label Scheme (Binary)
| 값 | 의미 |
|----|------|
| 0  | No-Cloud (Clear / Snow / Water) |
| 1  | Cloud (Cloud + Shadow + Cirrus + Dilated) |
| 255 | No-Data (ignore_index) |

---

## 1. 학습 데이터(패치) 생성

모델 학습 전에 GeoTIFF 원본 영상들을 읽어 Zarr(`*.zarr`) 형태의 패치 데이터로 변환해야 합니다.

### 스크립트 실행 방법

**기본 데이터 경로를 사용하는 경우**:
```bash
# 학습용(Train) 데이터 패치 생성
python make_landsat_data.py --mode train

# 검증용(Validation) 데이터 패치 생성
python make_landsat_data.py --mode test
```

**커스텀 데이터 경로를 사용하는 경우**:
```bash
python make_landsat_data.py --mode train --path /path/to/your/landsat/scenes
```

> **참고**: 원본 씬 폴더에는 `*_B1.TIF` ~ `*_B7.TIF` 및 라벨용 `*_QA_PIXEL.TIF`가 필수적으로 있어야 합니다. `*_B9.TIF`(Cirrus 밴드)는 선택 사항이며, 없을 경우 0으로 자동 채워집니다.

### Zarr Patch Format
```
spectral/    → B1–B7 + B9 (uint16, shape H×W×8)
rgb/         → OpenCV RGB, 퍼센타일 정규화 (float32, H×W×3)
hsv/         → OpenCV HSV (float32, H×W×3)
sobel/       → Sobel X / Y / Magnitude (float32, H×W×3)
qa_label/    → 바이너리 라벨 0/1/255 (uint8, H×W)
```

---

## 2. 모델 학습 방법

데이터 생성이 완료되면 4-Stage Self-Training 파이프라인을 구동합니다.

### 전체 파이프라인 일괄 실행 (`pipeline.sh`)

```bash
./pipeline.sh <실험명(exp_name)> [입력모드(input_mode)] [사용할_GPU_ID]
```

**실행 예시:**
```bash
# 실험명 weddell_exp1, 입력모드 swirndsi (기본값)로 GPU 0번과 1번 사용
./pipeline.sh weddell_exp1 "swirndsi" "0 1"

# 실험명 exp2, all_derived 모드로 GPU 0번 사용
./pipeline.sh exp2 "all_derived" "0"
```

> Stage 0 (QA 바이너리 라벨 학습)부터 Stage 3 (가장 큰 네트워크)까지 pseudo-label 생성과 학습이 순차적으로 자동 진행됩니다.

### 수동 실행 예시
```bash
# stage 0 학습
python train.py -e my_experiment -st 0 -ip swirndsi -gpu 0

# pseudo-label 생성 후 다음 stage 학습
python label_generation.py -e my_experiment -st 0
python train.py -e my_experiment -st 1 -ip swirndsi -gpu 0
```

---

## 3. 주요 조작 파라미터 (`train.py`)

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

커스텀 조합:
```bash
python train.py --inp_mode custom --bands B2 B3 B4 B5 B6 --indices NDSI NDWI
```

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
# 다중 GPU 사용 예시 (DataParallel)
python train.py -e my_exp -st 0 -ip swirndsi -gpu 0 1
```

---

## Training Pipeline

| Stage | 네트워크 | 라벨 소스 | 데이터 |
|-------|---------|-----------|-------|
| 0 | depth=5, filters=16 | QA_PIXEL (binary) | stage_0.txt |
| 1 | depth=5, filters=32 | pseudo-label | stage_0+1.txt |
| 2 | depth=6, filters=24 | pseudo-label | stage_0+1+2.txt |
| 3 | depth=6, filters=32 | pseudo-label | 전체 |
