# Landsat 8 구름 탐지 — 학습 파이프라인 상세 설명

## 목차
1. [전체 개요](#1-전체-개요)
2. [데이터 및 라벨 체계](#2-데이터-및-라벨-체계)
3. [모델 구조: U-Net](#3-모델-구조-u-net)
4. [입력 채널 구성](#4-입력-채널-구성)
5. [Stage별 학습 흐름](#5-stage별-학습-흐름)
6. [손실 함수와 클래스 가중치 (MFB)](#6-손실-함수와-클래스-가중치-mfb)
7. [검증 및 Early Stopping](#7-검증-및-early-stopping)
8. [데이터 증강](#8-데이터-증강)
9. [Pseudo-label 생성](#9-pseudo-label-생성)
10. [전체 흐름 예시](#10-전체-흐름-예시)

---

## 1. 전체 개요

이 파이프라인은 **Self-Training(자기학습)** 방식의 반지도 학습입니다.

핵심 아이디어: 처음에는 자동 생성된 노이즈 라벨(QA_PIXEL)로 작은 모델을 학습시키고, 그 모델이 생성한 더 나은 예측(pseudo-label)으로 점점 큰 모델을 학습시켜 성능을 반복적으로 높입니다.

```
QA_PIXEL 라벨 (노이즈 있음)
        ↓
  Stage 0 학습 (작은 모델)
        ↓
  Stage 0 모델로 pseudo-label 생성
        ↓
  Stage 1 학습 (더 큰 모델 + 더 많은 데이터)
        ↓
  Stage 1 모델로 pseudo-label 재생성
        ↓
  Stage 2, 3 반복...
        ↓
  최종 모델 (stage 3)
```

**Stage별 네트워크 크기**:

| Stage | Depth | Start Filters | 데이터 비율 |
|-------|-------|--------------|------------|
| 0     | 5     | 16           | 25% (QA_PIXEL 라벨) |
| 1     | 5     | 32           | 50% (QA_PIXEL + pseudo) |
| 2     | 6     | 24           | 75% (QA_PIXEL + pseudo) |
| 3     | 6     | 32           | 100% (QA_PIXEL + pseudo) |

---

## 2. 데이터 및 라벨 체계

### 패치 형식
각 패치는 `256×256` 크기의 Zarr 디렉토리로 저장됩니다:

```
LC08_L1GT_..._PATCH42.zarr/
  spectral/   (256, 256, 8)  uint16  — B1~B7, B9 원시 DN값
  rgb/        (256, 256, 3)  float32 — 백분위 정규화된 RGB
  hsv/        (256, 256, 3)  float32 — Hue, Saturation, Value
  sobel/      (256, 256, 3)  float32 — Sobel X, Y, Magnitude
  label/      (256, 256)     uint8   — 이진 라벨
  pseudo_label/ (256, 256)   uint8   — stage 1+ 에서 추가됨
```

### 라벨 값 의미

```
0   → no-cloud  (맑은 하늘, water, snow/ice)
1   → cloud     (구름, 구름 그림자)
255 → ignore    (fill pixel, 미라벨 영역)
```

`255`는 손실 함수에서 완전히 무시됩니다 (`ignore_index=255`).

### 훈련 데이터 분할
첫 실행 시 `split_data()`가 TRAIN_ZARR 패치들을 랜덤하게 4개 단계로 나눕니다:

```
전체 5,118개 패치
├── stage_0.txt  (25%)  → ~1,279개  QA_PIXEL 라벨 사용
├── stage_1.txt  (25%)  → ~1,279개  pseudo-label 생성 대상
├── stage_2.txt  (25%)  → ~1,279개  pseudo-label 생성 대상
└── stage_3.txt  (25%)  → ~1,279개  pseudo-label 생성 대상
```

---

## 3. 모델 구조: U-Net

U-Net은 인코더-디코더 구조로, **인코더에서 추출한 특징을 스킵 연결(skip connection)로 디코더에 전달**하여 공간 정보를 보존합니다.

### 전체 구조도 (Stage 0 예시: depth=5, filters=16)

```
입력 (B, 7, 258, 258)   ← 256×256 + 1픽셀 패딩
       │
  ┌────▼────┐
  │Encoder 0│  DoubleConv(7→16)  → MaxPool  → skip_0: (B,16,258,258)
  └─────────┘
       │
  ┌────▼────┐
  │Encoder 1│  DoubleConv(16→32) → MaxPool  → skip_1: (B,32,129,129)
  └─────────┘
       │
  ┌────▼────┐
  │Encoder 2│  DoubleConv(32→64) → MaxPool  → skip_2: (B,64,64,64)
  └─────────┘
       │
  ┌────▼────┐
  │Encoder 3│  DoubleConv(64→128)→ MaxPool  → skip_3: (B,128,32,32)
  └─────────┘
       │
  ┌────▼────┐
  │Encoder 4│  DoubleConv(128→256) 풀링 없음 (bottleneck)
  └─────────┘
       │
  ┌────▼────┐
  │Decoder 3│  UpConv(256→128) + Concat(skip_3) → DoubleConv
  └─────────┘
       │
  ┌────▼────┐
  │Decoder 2│  UpConv(128→64)  + Concat(skip_2) → DoubleConv
  └─────────┘
       │
  ┌────▼────┐
  │Decoder 1│  UpConv(64→32)   + Concat(skip_1) → DoubleConv
  └─────────┘
       │
  ┌────▼────┐
  │Decoder 0│  UpConv(32→16)   + Concat(skip_0) → DoubleConv
  └─────────┘
       │
  Conv1x1 → (B, 2, 258, 258)   ← 2 클래스 (no-cloud, cloud)
```

### DoubleConv 블록 (모든 인코더/디코더의 기본 단위)

```
입력
  → Conv3x3 → BatchNorm → ReLU
  → Conv3x3 → BatchNorm → ReLU
  → Dropout2d(p=0.25)   ← 학습 중에만 작동, 채널 단위로 0으로 만들어 과적합 방지
출력
```

### 단계별 모델 크기 변화

| Stage | Depth | Filters | Bottleneck 채널 수 | 파라미터 수 (대략) |
|-------|-------|---------|-------------------|------------------|
| 0     | 5     | 16      | 256               | ~1.2M            |
| 1     | 5     | 32      | 512               | ~4.8M            |
| 2     | 6     | 24      | 768               | ~8.1M            |
| 3     | 6     | 32      | 1024              | ~14.3M           |

Stage가 올라갈수록 더 넓고 깊은 네트워크로 교체됩니다.  
이전 stage의 가중치를 이어받지 않고 **매 stage마다 새로 초기화**합니다.

---

## 4. 입력 채널 구성

패치 파일에는 17채널이 저장되어 있지만, 모델에는 선택된 채널만 입력됩니다.

### `swirndsi` 모드 (기본값, 7채널)

```
[B2 Blue, B3 Green, B4 Red, B5 NIR, B6 SWIR1, B7 SWIR2, NDSI]
```

**NDSI (Normalized Difference Snow Index)**:

```
NDSI = (Green - SWIR1) / (Green + SWIR1)
```

- 눈/얼음: NDSI ≈ +0.8 (Green 강함, SWIR1 흡수)
- 구름: NDSI ≈ +0.1~0.3 (둘 다 높음)
- 육지: NDSI ≈ -0.1~0.2

NDSI를 추가하면 **눈/얼음과 구름을 구별**하는 핵심 신호가 됩니다.

### 다른 모드 요약

| 모드            | 채널 수 | 특징 |
|----------------|--------|------|
| `swirndsi`     | 7      | 기본값. 눈/얼음 구별 강점 |
| `cirrus_ndsi`  | 8      | Cirrus(권운) 밴드 추가 — 얇은 구름 탐지에 유리 |
| `all_derived`  | 17     | 전체 스펙트럼 + RGB/HSV/Sobel |
| `rgb`          | 3      | 가시광선만 (성능 낮음) |

---

## 5. Stage별 학습 흐름

### Stage 0

```python
python train.py -e exp_stage0 -st 0 -ip swirndsi -gpu 0
```

- **데이터**: stage_0.txt (~1,279개 패치)
- **라벨**: `store['label']` — QA_PIXEL 기반 binary 라벨
- **네트워크**: depth=5, filters=16 (가장 작음)
- **목표**: QA_PIXEL 라벨의 노이즈를 어느 정도 견디며 기본 구름 탐지 능력 습득

```
Epoch 1/400
  Train: 1,279 패치 / 32 batch_size = 40 iteration
         → 40번 가중치 업데이트
  Valid: 1,174 val 패치 모두 평가
         → mIoU 계산, early stopping 체크
...
Early Stopping (patience=10 연속 개선 없으면 종료)
→ model_best.pth 저장
```

### Pseudo-label 생성 (Stage 0→1)

```python
python label_generation.py -e exp_stage0 -st 1 -gpu 0
```

- stage_0 모델(`model_best.pth`)을 불러와 stage_1.txt 패치에 추론
- 예측 확률 최댓값 < 0.4인 픽셀은 `255`(ignore)로 처리 — 불확실한 픽셀은 학습에서 제외
- 결과를 각 패치의 `pseudo_label` 배열로 저장

```
패치 하나 예시:
  입력: (7, 258, 258) 텐서
  출력: softmax → argmax → pseudo_label (256, 256)
  
  예) 모델 출력 확률:
    픽셀 (100, 100):  no-cloud=0.92, cloud=0.08  → 0 저장 (확실)
    픽셀 (150, 200):  no-cloud=0.61, cloud=0.39  → 255 저장 (불확실, ignore)
    픽셀 (200, 100):  no-cloud=0.05, cloud=0.95  → 1 저장 (확실)
```

### Stage 1

```python
python train.py -e exp_stage1 -st 1 -ip swirndsi -gpu 0
```

- **데이터**: stage_0.txt + stage_1.txt (~2,558개 패치)
  - stage_0 패치: `label` (QA_PIXEL) 사용
  - stage_1 패치: `pseudo_label` 사용
- **네트워크**: depth=5, filters=32 (stage 0보다 4배 파라미터)

### Stage 2, 3도 동일한 패턴

```
Stage 2: stage_0~2 데이터 (3,837패치), depth=6, filters=24
Stage 3: 전체 데이터 (5,116패치), depth=6, filters=32 (최대)
```

---

## 6. 손실 함수와 클래스 가중치 (MFB)

### 손실 함수: Weighted Cross-Entropy

```python
loss = F.cross_entropy(output, labels, weight=w, ignore_index=255)
```

`ignore_index=255`이므로 fill 픽셀과 미라벨 픽셀은 손실 계산에서 완전히 제외됩니다.

### Median Frequency Balancing (MFB)

구름은 no-cloud보다 훨씬 적게 등장하는 불균형 데이터입니다.  
MFB는 희귀 클래스에 더 높은 가중치를 부여해 모델이 다수 클래스에 편향되지 않도록 합니다.

```
train 데이터 전체 순회 → 클래스별 픽셀 빈도 계산

예시:
  no-cloud(0): 65.6%
  cloud(1):    34.4%

  freq = [0.656, 0.344]
  median_freq = 0.500  (두 값의 중앙값)

  weight[0] = 0.500 / 0.656 = 0.76  ← 흔한 클래스 → 가중치 낮음
  weight[1] = 0.500 / 0.344 = 1.45  ← 희귀 클래스 → 가중치 높음
```

결과적으로 구름 픽셀을 틀리면 no-cloud를 틀릴 때보다 1.45/0.76 ≈ **1.9배** 더 큰 손실을 받습니다.

---

## 7. 검증 및 Early Stopping

### 검증 지표: mIoU

각 epoch 종료 후 validation 패치 전체로 평가합니다.

```
Confusion Matrix (2×2):
          예측 no-cloud  예측 cloud
실제 no-cloud    TN           FP
실제 cloud       FN           TP

IoU(no-cloud) = TN / (TN + FP + FN)
IoU(cloud)    = TP / (TP + FP + FN)
mIoU          = (IoU_0 + IoU_1) / 2
```

### Early Stopping 조건

```
patience = 10

for epoch in 1..400:
    학습 + 검증 → mIoU 기록

    epoch > 10이면:
        최근 10 epoch mIoU 이동평균 계산
        이동평균이 역대 최고보다 낮으면 patience_counter += 1
        patience_counter == 10이면 → 학습 종료

최고 mIoU를 기록한 epoch의 모델 → model_best.pth
```

**예시 흐름**:
```
Epoch 15: mIoU=0.42  이동평균=0.38  → 최고 갱신
Epoch 16: mIoU=0.41  이동평균=0.40  → 최고 갱신
Epoch 17: mIoU=0.40  이동평균=0.40  → 동일, counter=1
...
Epoch 26: mIoU=0.39  이동평균=0.39  → counter=10 → 종료
→ model_16.pth를 model_best.pth로 복사
```

---

## 8. 데이터 증강

Stage 1 이상에서 `--aug` 플래그 활성화 시 적용됩니다.  
5가지 변환이 각각 50% 확률로 독립적으로 적용됩니다.

| 변환 | 설명 |
|------|------|
| HorizontalFlip | 좌우 반전 |
| VerticalFlip   | 상하 반전 |
| Rotate90       | 90°/180°/270° 중 랜덤 회전 |
| CutOut         | 이미지의 10~30% 크기 사각형 영역을 0으로 채움. 해당 영역 라벨은 255(ignore) |
| ZoomIn         | 1.0~1.5× 랜덤 확대 후 원래 크기로 리사이즈 |

이미지와 라벨은 **항상 같은 변환을 적용**합니다 (같은 파라미터로 동시에 처리).

---

## 9. Pseudo-label 생성

```python
# label_generation.py 핵심 로직 (model.py encode_label)

softmax_out = softmax(network_output)          # (B, 2, H, W) 확률
predicted   = argmax(softmax_out, dim=1)        # (B, H, W) 0 또는 1
confidence  = max(softmax_out, dim=1)           # (B, H, W) 최대 확률값

predicted[confidence < 0.4] = 255              # 불확실 → ignore
```

**threshold=0.4의 의미**:
- confidence가 0.4 미만 = 두 클래스 확률이 각각 0.4~0.6 사이 = 모델이 어떤 클래스인지 확신 못함
- 이런 픽셀을 pseudo-label에 포함시키면 오히려 다음 stage 학습에 노이즈가 됨
- 따라서 ignore로 처리해 손실 계산에서 제외

**QA_PIXEL nodata 전파**:
```python
qa_label = store['label'][:]
label_np[qa_label == 255] = 255   # 원래 fill 픽셀은 무조건 ignore 유지
```

---

## 10. 전체 흐름 예시

실제 수치를 기반으로 한 전체 흐름입니다.

```
[데이터 준비]
TRAIN_ZARR: 5,118 패치 (256×256, QA_PIXEL 라벨)
VALID_ZARR: 1,174 패치 (수동 cloud 라벨)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력: 7채널 (B2~B7 + NDSI)
모델: UNet(depth=5, filters=16) — 약 1.2M 파라미터
데이터: stage_0.txt (1,279 패치)
라벨: QA_PIXEL binary

Epoch 1: Train(40 iter) → Valid(37 iter) → mIoU=0.35
Epoch 2: Train → Valid → mIoU=0.41
...
Epoch 23: mIoU 이동평균 정체 → Early Stop
→ model_best.pth (epoch 13, mIoU=0.52) 저장

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PSEUDO-LABEL 생성 (Stage 0 → Stage 1용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
stage_1.txt 1,279 패치에 model_best.pth로 추론
  → 확률 ≥ 0.4인 픽셀만 pseudo_label 저장
  → 불확실 픽셀은 255(ignore)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모델: UNet(depth=5, filters=32) — 새로 초기화, 약 4.8M 파라미터
데이터: stage_0 (1,279 QA) + stage_1 (1,279 pseudo) = 2,558 패치
  → stage_0: label 사용
  → stage_1: pseudo_label 사용

...→ model_best.pth (mIoU 향상됨)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2, 3: 동일 패턴 반복
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
최종 결과: exp_data/swirndsi_trial2_stage3/model/model_best.pth
```

---

## 실행 명령어 요약

```bash
cd /home/pyuncb/src

# 전체 파이프라인 한 번에 실행
nohup ./pipeline.sh swirndsi_trial2 swirndsi "0" > pipeline_trial2.log 2>&1 &

# 로그 확인
tail -f pipeline_trial2.log

# 또는 단계별 수동 실행
python train.py -e swirndsi_trial2_stage0 -st 0 -ip swirndsi -gpu 0
python label_generation.py -e swirndsi_trial2_stage0 -st 1 -gpu 0
python train.py -e swirndsi_trial2_stage1 -st 1 -ip swirndsi -gpu 0
python label_generation.py -e swirndsi_trial2_stage1 -st 2 -gpu 0
python train.py -e swirndsi_trial2_stage2 -st 2 -ip swirndsi -gpu 0
python label_generation.py -e swirndsi_trial2_stage2 -st 3 -gpu 0
python train.py -e swirndsi_trial2_stage3 -st 3 -ip swirndsi -gpu 0
```
