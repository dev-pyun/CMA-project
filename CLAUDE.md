# CLAUDE.md

## Project Overview
Landsat 8 위성 영상에서 Cloud / Cloud Shadow / Snow / Water / Clear-Sky Land를 분류하는 **semi-supervised self-training segmentation 파이프라인**.
- UNet 기반 4-stage progressive 학습 (stage 0: QA_PIXEL 라벨 → stage 1–3: pseudo-label)
- 입력: Landsat 8 Collection 2 Level-1 GeoTIFF (B1–B7, QA_PIXEL)
- 출력: 6-class segmentation map (No-Data / Clear / Cloud / Shadow / Snow / Water)

## Directory Structure
```
src/
├── train.py              # 메인 학습 스크립트
├── predict.py            # 추론 스크립트
├── label_generation.py   # Pseudo-label 생성 (stage N → N+1)
├── make_landsat_data.py  # GeoTIFF → HDF5 변환 진입점
├── dataset/
│   ├── patch_dataset.py  # PyTorch Dataset (HDF5 patch 로딩)
│   ├── network_input.py  # Band 선택 + 분광 지수 계산
│   └── transforms.py     # Data augmentation
├── network/
│   ├── model.py          # UNet wrapper + 학습/검증 로직
│   └── unet.py           # UNet 구현
├── utils/
│   ├── split_scene.py    # GeoTIFF → HDF5 변환 핵심 로직
│   ├── qa_pixel_mapping.py  # QA_PIXEL bitmask → 6-class 라벨
│   ├── dir_paths.py      # 경로 상수 (여기서 소스 경로 관리)
│   ├── experiment.py     # 실험 관리 + 체크포인트
│   ├── MFB.py            # Median Frequency Balancing (클래스 가중치)
│   ├── metrics.py        # Accuracy / IoU 계산
│   ├── csv_logger.py     # 학습 메트릭 CSV 저장
│   └── join_predictions.py  # 예측 후처리
├── data/
│   ├── TRAIN_H5/         # 학습용 HDF5 패치 (256×256)
│   └── VALIDATION_H5/    # 검증용 HDF5 패치
└── exp_data/             # 실험 결과 (모델, 로그)
```

## Data Source
- **Weddell Sea 원본 데이터**: `/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/`
  - 구조: `{year}/{month}/{date}/{scene_id}/` (예: `2020/01/20200101/LC08_L1GT_.../`)
  - 데이터 복사 없이 직접 읽어 HDF5 패치 생성
- **HDF5 출력**: `src/data/TRAIN_H5/`

## Setup
```bash
# 의존성 설치 (rasterio, h5py, torch, tqdm, numpy, opencv)
pip install rasterio h5py torch tqdm numpy opencv-python-headless

# HDF5 패치 생성 (Weddell Sea 소스에서 직접 읽기)
python make_landsat_data.py --mode train --source weddell

# 학습 (stage 0)
python train.py -e my_experiment -st 0 -ip swirndsi -gpu 0

# Pseudo-label 생성 후 다음 stage 학습
python label_generation.py -e my_experiment -st 0
python train.py -e my_experiment -st 1 -ip swirndsi -gpu 0
```

## HDF5 Patch Format
```
data[:, :, 0:7]  → B1–B7 (uint16, DN 값, 로딩 시 ÷10000으로 정규화)
data[:, :, 7]    → QA_PIXEL 라벨 (0–5)
data[:, :, 9]    → Pseudo-label (label_generation.py가 추가)
```

## Training Pipeline
| Stage | 네트워크 | 라벨 소스 | 데이터 |
|-------|---------|-----------|-------|
| 0 | depth=5, filters=16 | QA_PIXEL | stage_0.txt |
| 1 | depth=5, filters=32 | pseudo-label | stage_0+1.txt |
| 2 | depth=6, filters=24 | pseudo-label | stage_0+1+2.txt |
| 3 | depth=6, filters=32 | pseudo-label | 전체 |

## Code Style
- Python, 기능별 모듈화 (파일당 단일 책임)
- Type hint 사용
- 경로 상수는 `utils/dir_paths.py`에서만 관리

## 주의사항
- 많은 부분을 수정해야한다면 반드시 나에게 물어보고 진행해.
- 하나의 파일에 코드를 다 넣지 말고, 기능별로 모듈화해
- 요청이 명확하지 않을 때 추론 및 실행하지말고, 내 설명을 제대로 이해했는지 확인해.
- 수정을 진행할 때마다 반드시 수정사항을 기록해줘.
- 코드 실행 결과가 예상과 다를때, 코드 실행 과정과 예상 결과를 나에게 보여줘.
- 앞으로 모든 진행사항과 수정사항이 생길때마다 반드시 그 진행사항 및 수정사항을 /home/pyuncb/src/WALKTHROUGH.md에 기록해줘. 
- 모든 코드 실행은 ssh가 끊겨도 백그라운드에서 실행되도록 해줘. (nohup / tmux 등 사용)
- 원본 논문은 src/Self-trained model for cloud, shadow and snow detection in sentinel-2 images of snow- and ice- covered regions.pdf에 있으니 이걸 참고해