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
├── test_pipeline.py      # 3-class 파이프라인 검증 스크립트 (16개 항목)
├── update_train_labels.py  # TRAIN_ZARR label binary→3-class in-place 업데이트
├── compute_val_confusion.py  # Validation 혼동 행렬 평가 (5개 실험 병렬)
├── run_val_confusion.sh  # compute_val_confusion.py nohup 실행 래퍼
├── compare_scene.py      # 씬 전체 Fmask / 모델 예측 / GT 3-panel 비교
├── compare_stages.sh     # stage 0~3 비교 시각화 파이프라인
├── vis_pipeline.sh       # 씬 하나에 대해 stage별 예측 이미지 일괄 생성
├── visualize_comparison.py  # 개별 zarr 패치 3-panel 비교
├── inspect_zarr.py       # 개별 zarr 패치 내용 확인
├── vis_cv_features.py    # 씬별 42가지 CV 피처 그리드 시각화
├── vis_pca.py            # PCA 8개 성분 시각화 + 밴드 상관관계
├── vis_pca_transfer.py   # Scene A PCA를 Scene B에 전이 적용 시각화
├── diagnose_label_offset.py  # zarr 패치 label 좌표 정합성 진단
├── dataset/
│   ├── patch_dataset.py  # PyTorch Dataset (Zarr patch 로딩)
│   ├── network_input.py  # Band 선택 + 분광 지수 계산
│   └── transforms.py     # Data augmentation
├── network/
│   ├── model.py          # UNet wrapper + 학습/검증 로직
│   └── unet.py           # UNet 구현
├── utils/
│   ├── split_scene.py    # GeoTIFF → Zarr 패치 변환 핵심 로직
│   ├── qa_pixel_mapping.py  # QA_PIXEL bitmask → 3-class 라벨
│   ├── scene_inference.py   # 씬 데이터 로드, 타일 추론, CM 누적 유틸
│   ├── compute_global_stats.py  # Weddell Sea 전체 씬 global mean/std/PCA 계산
│   ├── dir_paths.py      # 경로 상수 (여기서 소스 경로 관리)
│   ├── experiment.py     # 실험 관리 + 체크포인트
│   ├── MFB.py            # Median Frequency Balancing (sqrt-MFB)
│   ├── metrics.py        # Accuracy / per-class IoU 계산
│   ├── csv_logger.py     # 학습 메트릭 CSV 저장
│   └── join_predictions.py  # 예측 후처리
├── label_code/           # napari 기반 수동 라벨링 도구 (label_code/README.md 참고)
├── data/
│   ├── TRAIN_ZARR/       # 학습용 Zarr 패치 (258×258 TOA)
│   ├── VALIDATION_ZARR/  # 검증용 Zarr 패치 (수동 라벨 기반)
│   ├── TEST_ZARR/        # 테스트용 Zarr 패치
│   ├── global_spectral_stats.npz  # 전체 씬 밴드별 global mean/std
│   └── global_pca.npz    # Global PCA eigenvectors (8 components)
├── val_confusion/        # 모델 평가 혼동 행렬 출력
└── exp_data/             # 실험 결과 (모델, 로그)
```

## Data Source
- **Weddell Sea 원본 데이터**: `/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/`
  - 구조: `{year}/{month}/{date}/{scene_id}/` (예: `2020/01/20200101/LC08_L1GT_.../`)
  - 데이터 복사 없이 직접 읽어 Zarr 패치 생성

## Label Scheme

### 학습 파이프라인 / Zarr patch 저장 형식 (3-class)
| 값 | 의미 | loss |
|----|------|------|
| 0  | No-Cloud (Clear / Snow / Water) | 포함 |
| 1  | Cloud (Cloud + Cirrus + Dilated) | 포함 |
| 2  | Cloud Shadow | 포함 |
| 255 | No-Data / ignore | **무시** (ignore_index=255) |

### 수동 라벨링 (napari, label_scene.py 출력)
| 값 | 의미 | napari 키 | patch 저장 값 |
|----|------|----------|--------------|
| 0  | 미라벨 (napari 기본값) | `0` | 255 (ignore) |
| 1  | water | `1` | 0 (no-cloud) |
| 2  | snow / ice | `2` | 0 (no-cloud) |
| 3  | cloud shadow (**명확한 경우만**) | `3` | 2 (shadow) |
| 4  | cloud (opaque + cirrus + dilated) | `4` | 1 (cloud) |
| 255 | 센서 fill (자동 마킹) | — | 255 (ignore) |

remap 규칙 (scene_to_patches.py): `{1,2}→0`, `{3}→2`, `{4}→1`, `{0,255}→255`

---

## 1. 학습 데이터(패치) 생성

모델 학습 전에 GeoTIFF 원본 영상들을 읽어 Zarr(`*.zarr`) 형태의 패치 데이터로 변환해야 합니다.

### 스크립트 실행 방법

```bash
# 학습용(Train) 데이터 패치 생성
nohup conda run --no-capture-output -n remote python -u make_landsat_data.py --mode train \
    > logs/make_train_zarr.log 2>&1 &

# 검증용(Validation) 데이터 패치 생성
nohup conda run --no-capture-output -n remote python -u make_landsat_data.py --mode test \
    > logs/make_val_zarr.log 2>&1 &
```

> **참고**: 원본 씬 폴더에는 `*_B1.TIF` ~ `*_B7.TIF`, `*_QA_PIXEL.TIF`, `*_MTL.json`이 필수입니다. MTL은 TOA Reflectance 변환에 사용됩니다. `*_B9.TIF`(Cirrus)는 선택 사항이며, 없으면 0으로 채워집니다.

### Zarr Patch Format
```
spectral/    → B1–B7 + B9, TOA Reflectance × 10000 (uint16, 258×258×8)
             ※ 실제 활용 크기 256×256, ±1px border는 인접 픽셀로 채워짐 (경계 아티팩트 방지)
rgb/         → OpenCV RGB 퍼센타일 정규화 (float32, 256×256×3)
hsv/         → OpenCV HSV (float32, 256×256×3)
sobel/       → Sobel X / Y / Magnitude (float32, 256×256×3)
qa_label/    → 3-class 라벨 0/1/2/255 (uint8, 256×256)
```

---

## 2. 모델 학습 방법

### 전체 파이프라인 일괄 실행 (`pipeline.sh`)

```bash
./pipeline.sh <실험명(exp_name)> [입력모드(input_mode)] [사용할_GPU_ID] [클래스_수(num_classes)]
```

**실행 예시:**
```bash
# 기본 (3-class)
nohup conda run --no-capture-output -n remote bash pipeline.sh \
    weddell_exp1 swirndsi "0 1" 3 > logs/pipeline_weddell_exp1.log 2>&1 &

# 2-class (shadow 무시)
nohup conda run --no-capture-output -n remote bash pipeline.sh \
    exp_ndsi679 ndsi679 "0" 2 > logs/pipeline_ndsi679.log 2>&1 &
```

> Stage 0 (QA 3-class 라벨) → Stage 3 (가장 큰 네트워크)까지 pseudo-label 생성과 학습이 순차 진행됩니다.

### 수동 실행 예시
```bash
conda run -n remote python train.py -e my_experiment -st 0 -ip swirndsi -gpu 0
conda run -n remote python label_generation.py -e my_experiment -st 0
conda run -n remote python train.py -e my_experiment -st 1 -ip swirndsi -gpu 0
```

### 모든 실행 nohup 패턴 (SSH 끊겨도 유지)
```bash
# Python 스크립트
nohup conda run --no-capture-output -n remote python -u script.py ... > logs/xxx.log 2>&1 &

# Bash 스크립트
nohup conda run --no-capture-output -n remote bash script.sh ... > logs/xxx.log 2>&1 &

# 진행 확인
tail -f logs/xxx.log
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
#   --init_cfmask: cloud + shadow 영역을 CFMask 기반으로 미리 초기화
python label_scene.py \
    --prepared_dir prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --init_cfmask

# Step 3. 256×256 patch 분할 (VALIDATION_ZARR 또는 TEST_ZARR에 저장)
python scene_to_patches.py \
    --scene_dir  /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --split val
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

| 프리셋 | 채널 | 채널 구성 |
|--------|:---:|---------|
| `swirndsi` (기본) | 7 | B2–B7 + NDSI |
| `cirrus_ndsi` | 8 | B2–B7 + B9(Cirrus) + NDSI |
| `cirrus_ndsindwi` | 9 | B2–B7 + B9 + NDSI + NDWI |
| `swirndsi_pca3` | 10 | B2–B7 + NDSI + global PC1–PC3 |
| `swirndsindwi_pca3` | 11 | B2–B7 + NDSI + NDWI + global PC1–PC3 |
| `ndsi679` | 4 | NDSI(B5,B6) + B6 + B7 + B9 — 2-class cloud-only 실험용 |
| `all_derived` | 17 | 스펙트럼 + RGB + HSV + Sobel |
| `rgb_hsv` | 6 | RGB + HSV |
| `swir_sobel` | 5 | B5–B6 + Sobel |
| `all_cirrus` | 8 | B1–B7 + B9 |
| `rgb` | 3 | B2–B4 |

> `swirndsi_pca3` / `swirndsindwi_pca3` 사용 시 `data/global_pca.npz`가 반드시 존재해야 합니다.

### B. 학습 하이퍼파라미터

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `-lr` | 학습률 | `1e-6` |
| `-bs` | 배치 사이즈 | `64` |
| `-ep` | 최대 에폭 수 | `400` |
| `--seed` | 랜덤 시드 | `42` |
| `--num_classes` | 클래스 수 (2 또는 3) | `3` |
| `--no_dropout` | 드롭아웃 비활성화 | — |
| `--no_aug` | 데이터 증강 비활성화 | — |

### C. 네트워크 / 파이프라인

| 인자 | 설명 |
|------|------|
| `-st` | Self-training 스테이지 (0–3) |
| `--full` | 가장 큰 네트워크로 단일 지도학습 |

### D. 하드웨어
```bash
conda run -n remote python train.py -e my_exp -st 0 -ip swirndsi -gpu 0 1   # 다중 GPU
```

---

## Training Pipeline

| Stage | 네트워크 | 라벨 소스 | 데이터 |
|-------|---------|-----------|-------|
| 0 | depth=5, filters=16 | QA_PIXEL (3-class) | stage_0.txt |
| 1 | depth=5, filters=32 | pseudo-label | stage_1.txt (신규 추가분만) |
| 2 | depth=6, filters=24 | pseudo-label | stage_2.txt (신규 추가분만) |
| 3 | depth=6, filters=32 | pseudo-label | stage_3.txt (신규 추가분만) |

> label_generation.py는 현재 stage의 신규 데이터만 pseudo-label 생성 (이전 stage 데이터 재추론 불필요).

---

## 5. 결과 시각화

### 씬 전체 비교: Fmask vs 모델 예측 vs Ground Truth (`compare_scene.py`)

학습된 모델의 예측 결과를 씬 단위로 시각화합니다.
Fmask / 모델 예측 / Ground Truth를 씬별 폴더에 개별 PNG로 저장합니다.

```bash
cd /home/pyuncb/src

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

python compare_scene.py \
    --scene_dir  $WEDDELL/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path label_code/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --exp        swirndsi_trial2_stage0 \
    --stage      0 \
    --gpu        0 \
    --out        vis_output/
```

출력 구조:
```
vis_output/{scene_id}/
    fmask.png
    model_{exp_name}.png
    ground_truth.png   ← label_path 있을 때만
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--scene_dir` | 원본 Landsat 씬 디렉토리 | 필수 |
| `--label_path` | 수동 라벨 GeoTIFF 경로 | 선택 (train 씬은 생략 가능) |
| `--exp` | 실험 이름 (`exp_data/` 하위) | 필수 |
| `--stage` | Self-training 스테이지 | `3` |
| `--inp_mode` | 입력 모드 (생략 시 체크포인트에서 자동 감지) | `None` |
| `--gpu` | GPU ID | `0` |
| `--out` | 결과 PNG 저장 디렉토리 | `vis_output/` |

---

### Stage 비교 시각화 (`compare_stages.sh`)

동일 패치에 대해 stage 0~3 전부를 한 번에 비교합니다.

```bash
./compare_stages.sh <exp_base> [n_samples] [gpu] [label_dir] [min_gini] [inp_mode]

# 예시
./compare_stages.sh swirndsi_trial2          # 기본 5개 샘플, GPU 0
./compare_stages.sh swirndsi_trial2 8 0      # 8개 샘플
./compare_stages.sh exp_swirndsi_pca3 5 0 label_code/labels 0.1 swirndsi_pca3
```

출력: `vis_output/{scene_id}/{scene_id}_PATCH{n}_{exp}_stage{N}_comparison.png`  
FCI 이미지도 씬 폴더에 자동 복사됩니다.

---

### 패치 단위 비교: Fmask vs 모델 예측 vs Ground Truth (`visualize_comparison.py`)

```bash
cd /home/pyuncb/src

WEDDELL=/earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea

# 단일 패치
python visualize_comparison.py \
    --patch data/VALIDATION_ZARR/LC08_L1GT_188114_20201114_20210315_02_T2_PATCH5.zarr \
    --exp   swirndsi_trial2_stage0 \
    --gpu   0

# VALIDATION_ZARR 디렉토리에서 6개 랜덤 샘플 (Gini 0.1 이상 필터)
python visualize_comparison.py \
    --patch    data/VALIDATION_ZARR/ \
    --exp      swirndsi_trial2_stage0 \
    --sample   6 \
    --min_gini 0.1 \
    --gpu      0 \
    --out      vis_output/
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--patch` | zarr 패치 경로 또는 VALIDATION_ZARR 디렉토리 | 필수 |
| `--exp` | 실험 이름 (`exp_data/` 하위) | 필수 |
| `--label_dir` | 수동 라벨 TIF 디렉토리 | `label_code/labels` |
| `--inp_mode` | 입력 모드 | `swirndsi` |
| `--gpu` | GPU ID | `0` |
| `--out` | 결과 PNG 저장 디렉토리 | `vis_output/` |
| `--sample` | 디렉토리 지정 시 랜덤 샘플 수 | — |
| `--min_gini` | 최소 Gini impurity 필터 (0=없음, 권장 0.1~0.3) | `0.0` |

---

### 패치 단위 검사 (`inspect_zarr.py`)

```bash
# 단일 패치 내용 출력 (텍스트)
python inspect_zarr.py data/VALIDATION_ZARR/LC08_L1GT_188114_..._PATCH5.zarr

# 시각화 이미지 저장
python inspect_zarr.py data/VALIDATION_ZARR/LC08_L1GT_188114_..._PATCH5.zarr --save

# 여러 패치 랜덤 샘플링
python inspect_zarr.py data/VALIDATION_ZARR/ --sample 9 --save --out output_vis/
```

---

## 6. Validation 모델 평가 (`compute_val_confusion.py`)

GT 라벨링된 8개 씬에 대해 학습된 모든 모델의 혼동 행렬 및 OA를 계산합니다.
5개 실험(×4 stage)을 멀티프로세스로 병렬 평가합니다.

```bash
cd /home/pyuncb/src
bash run_val_confusion.sh
# 또는
nohup /home/pyuncb/.conda/envs/cloud/bin/python compute_val_confusion.py \
    > logs/val_confusion.log 2>&1 &
```

출력 구조:
```
val_confusion/
├── {exp_base}_stage{N}/
│   ├── gt_vs_fmask.png          # GT vs Fmask (Fmask 품질 기준)
│   ├── gt_vs_{exp_base}.png     # GT vs 모델 (모델 성능)
│   └── fmask_vs_{exp_base}.png  # Fmask vs 모델 (Fmask 대비 개선도)
└── summary_oa.csv               # exp_name, oa_gt_fmask, oa_gt_model, oa_fmask_model
```

---

## 7. Global PCA / 통계 계산

```bash
# Pass 1 — 전체 씬 global mean/std 계산 (~45분)
nohup conda run --no-capture-output -n remote python -u utils/compute_global_stats.py \
    --root /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea \
    --max_size 300 > logs/global_stats.log 2>&1 &

# Pass 2 — global PCA fitting (~58분, Pass 1 완료 후)
nohup conda run --no-capture-output -n remote python -u utils/compute_global_stats.py \
    --pca_only > logs/global_pca.log 2>&1 &
```

출력:
- `data/global_spectral_stats.npz` — 밴드별 mean/std (표준화에 사용)
- `data/global_pca.npz` — PCA eigenvectors (PC1~96%, PC2~3.5%, PC3~0.6%)

`swirndsi_pca3` 등 PCA 기반 입력 모드 사용 시 필수입니다.

---

## GitHub push 설정 (SSH 키)

SSH 키 인증으로 설정되어 있습니다. 토큰 만료 없이 영구 사용 가능합니다.

### 현재 설정 확인

```bash
git remote -v
# origin  git@github.com:dev-pyun/CMA-project.git (fetch/push) 이어야 함
```

### 새 서버에서 설정할 때

```bash
# 1. SSH 키 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 공개키 확인 → GitHub Settings → SSH and GPG keys → New SSH key에 붙여넣기
cat ~/.ssh/id_ed25519.pub

# 3. remote URL을 SSH로 변경
git remote set-url origin git@github.com:dev-pyun/CMA-project.git

# 4. 연결 확인
ssh -T git@github.com
```
