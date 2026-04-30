# Cloud Masking 사이드 프로젝트 — 코드 사용법

남극 Landsat 8 cloud mask 모델 (Nambiar et al. 2022 self-training framework + L8 적응).
CFMask보다 나은 polar cloud mask를 self-training으로 학습.

---

## 환경 설정 (GPU 서버)

```bash
# 가상환경 권장
conda create -n cloud python=3.10 -y
conda activate cloud

# 의존성 설치
pip install -r requirements.txt

# PyTorch CUDA 빌드 (CUDA 버전 확인 후, 예: 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# SAM checkpoint (선택, vit_b 권장)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O ~/sam_vit_b.pth
```

---

## 워크플로우 — 한 씬 라벨링 (Phase 1: validation set)

### Step 1. 씬 준비 (CPU OK, ~5분)

```bash
python prepare_scene.py \
    --scene_dir /earth00_home/immj/Landsat/USGS/OLI_TIRS/lv1/Weddell_Sea/2020/11/20201114/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --out_dir   /home/pyuncb/src/label_code/prepared
```

출력 (`prepared/<scene_id>/`):
- `bands.tif` — (8, H, W) float32, B2~B7+B9 TOA reflectance + B10 brightness temp
- `fci.tif` — (3, H, W) uint8, FCI (B7/B5/B3) gamma 0.5
- `cfmask.tif` — (H, W) uint8, CFMask 5-class
- `meta.json` — CRS, sun angle, class 비율 등

### Step 2. napari 라벨링 (CPU/GPU, ~30분~1시간/씬)

```bash
python label_scene.py \
    --prepared_dir /home/pyuncb/src/label_code/prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --use_sam     # GPU 있으면 추가, CPU면 빼도 됨
```

**라벨링 단축키:**
- `1` : cloud 있음 (cloud + shadow + cirrus 모두 포함)
- `0` : cloud 없음 (clear + ice + water 모두 포함)
- `P` : Polygon mode (좌클릭으로 점, 우클릭 종료)
- `N` : Paint mode (브러시)
- `E` : Erase mode

**라벨링 원칙:**
- patch 경계 무시. 큰 scene 위에 자유롭게 폴리곤
- 확실한 영역만 → 모호한 cloud edge / thin cirrus 가장자리는 *미라벨(255)로 둠*
- CFMask가 *틀린 곳* 우선 → CFMask overlay 보면서 disagreement 영역 폴리곤
- cloud shadow는 cloud와 같은 클래스(1)로 라벨링

출력: `data/labels/<scene_id>_labels.tif`

### Step 3. 256×256 patch 자동 분할

```bash
python scene_to_patches.py \
    --prepared_dir /home/pyuncb/src/label_code/prepared/LC08_L1GT_188114_20201114_20210315_02_T2 \
    --label_path   /home/pyuncb/src/label_code/data/labels/LC08_L1GT_188114_20201114_20210315_02_T2_labels.tif \
    --patch_size 256 --stride 256
```

출력 (`data/patches/`):
- `val/<scene>_p{i}_{j}.h5` — 라벨된 픽셀 ≥ 5%
- `train_aux/<scene>_p{i}_{j}.h5` — 라벨된 픽셀 < 5% (보조 학습용)

각 patch HDF5:
- `/input` (8, 256, 256) float16 — 8-band 입력
- `/label` (256, 256) uint8 — 0=no-cloud, 1=cloud, 255=미라벨/fill
- `attrs` — scene_id, row/col, valid_label_frac, cfmask_*_frac 등

---

## 다음 단계 (이후 추가될 코드)

- `train_stage1.py` — Nambiar Stage 1: 작은 U-Net (16-start, depth 5), Fmask 노이즈 라벨로 학습
- `train_stage2.py` — Stage 1 teacher → Batch 2에 pseudo-label → 32-start U-Net
- `train_stage3.py` — Stage 2 teacher → Batch 3 pseudo-label → 24-start, depth 6
- `evaluate.py` — Test set IoU / F1 / mIoU, CFMask vs 자기 모델 비교

---

## 클래스 코드 요약

| Code | Class | 정의 | Loss |
|---|---|---|---|
| 0 | no-cloud | clear + ice + snow + water | 포함 |
| 1 | cloud | cloud + shadow + cirrus + dilated cloud | 포함 |
| 255 | 미라벨 / fill | 사람이 라벨 안 한 영역 또는 센서 결손 | **무시** (ignore_index=255) |

학습 시 loss는 클래스 0, 1만 계산. 255는 무시.

---

## 참고

- 이론 배경: `/home/immj/Labmeeting/project_cloud/MD/Project_Overview.md`
- Nambiar 논문: `/home/immj/Labmeeting/project_cloud/paper/Self-trained model for cloud, shadow and snow detection in sentinel-2 images of snow- and ice- covered regions.pdf`
