"""
Patch dataset loader for Landsat 8 Zarr patch files.

Loads patches created by utils/split_scene.py and serves them as
(input_tensor, labels, filename) tuples for training and validation.

Zarr patch layout (one .zarr directory per patch):
    spectral     (H+2, W+2, 8)  uint16  — B1–B7, B9 raw DN; 1px real border
    rgb          (H+2, W+2, 3)  float32 — percentile-normalised RGB ∈ [0,1]
    hsv          (H+2, W+2, 3)  float32 — H, S, V ∈ [0,1]
    sobel        (H+2, W+2, 3)  float32 — Sobel X, Y, Magnitude
    label        (H, W)         uint8   — binary: 0=no-cloud, 1=cloud, 255=no-data
    pseudo_label (H, W)         uint8   — added by label_generation.py (stage 1+)

H=W=256. Features are 258×258 with a real 1-pixel border from the scene.

Input tensor channel layout (17 channels):
    0–7   : B1–B7, B9  (float32, /10000 normalised)
    8–10  : RGB_R, RGB_G, RGB_B
    11–13 : HSV_H, HSV_S, HSV_V
    14–16 : Sobel_X, Sobel_Y, Sobel_Magnitude
"""

import glob
import logging
import os
import random

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

from dataset.transforms import VerticalFlip, HorizontalFlip, Rotate90, CutOut, ZoomIn

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

WORKERS = 16

N_SPECTRAL_BANDS = 8   # spectral channels in zarr 'spectral' array
N_DERIVED        = 9   # rgb(3) + hsv(3) + sobel(3)
N_TOTAL_CHANNELS = N_SPECTRAL_BANDS + N_DERIVED  # 17

NODATA_LABEL = 255  # ignored by cross-entropy loss


def set_seed(user_seed):
    """Set random seeds for reproducibility."""
    if user_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(user_seed)
        torch.manual_seed(user_seed)
        np.random.seed(user_seed)
        logger.info(f'Fixed seed: {user_seed}')
    else:
        logger.info('Training without fixed seed')


def check_data_split(train_path, reset=False):
    """Check if stage assignment files (stage_0.txt, etc.) already exist."""
    file_list = glob.glob(os.path.join(train_path, 'stage_*.txt'))
    if file_list:
        if reset:
            for stage_file in file_list:
                os.remove(stage_file)
        else:
            return True
    return False


def split_data(h5_folder, stage_0_ratio=0.25, stages=4):
    """
    Assign Zarr patch directories to different stages of the self-training pipeline.
    Stage 0 gets stage_0_ratio of the data; the rest is split evenly among stages 1–3.
    """
    h5_folder  = os.path.abspath(h5_folder)
    file_list  = glob.glob(os.path.join(h5_folder, '*.zarr'))
    n_files    = len(file_list)
    if not n_files:
        raise FileNotFoundError(f'No .zarr patches found in {h5_folder}')

    with open(os.path.join(h5_folder, 'stage_full.txt'), 'w') as f:
        for p in file_list:
            f.write(f'{p}\n')

    unlabelled_ratio = 1 - stage_0_ratio
    unlabelled_size  = int(n_files * unlabelled_ratio / (stages - 1))
    labeled_size     = n_files - (stages - 1) * unlabelled_size

    random.shuffle(file_list)

    start = 0
    for count in range(stages):
        stage_size = labeled_size if count == 0 else unlabelled_size
        end        = min(n_files, start + stage_size)
        with open(os.path.join(h5_folder, f'stage_{count}.txt'), 'w') as f:
            for p in file_list[start:end]:
                f.write(f'{p}\n')
        start = end


def setup_data(batch_size=1, mode='train', stage=0, path=None,
               full=False, aug=False, reset=False):
    """
    Set up the PyTorch DataLoader for a given mode and stage.

    Parameters
    ----------
    batch_size : int
    mode : str
        'train', 'test', 'label_gen', or 'predict'.
    stage : int
        Self-training stage (0–3).
    path : str
        Path to the Zarr patch directory.
    full : bool
        If True, use all data (supervised mode).
    aug : bool
        Whether to apply data augmentation.
    reset : bool
        Re-generate stage assignment files.
    """
    datasets = []
    shuffle  = (mode == 'train')

    if mode == 'train':
        if stage != 0 and reset:
            logger.warning('Stage data reset only in stage 0. Setting reset to False.')
            reset = False

        if not check_data_split(path, reset=reset):
            split_data(path)

        if full:
            file_path = os.path.join(path, 'stage_full.txt')
            with open(file_path) as fl:
                files_list = [line.rstrip() for line in fl]
            datasets.append(PatchDataset(mode, file_list=files_list, stage=0, aug=aug))
            logger.info(f'Full train set size: {len(files_list)}')
        else:
            for i in range(stage + 1):
                file_path = os.path.join(path, f'stage_{i}.txt')
                with open(file_path) as fl:
                    files_list = [line.rstrip() for line in fl]
                stage_aug = bool(stage) if aug else aug
                datasets.append(
                    PatchDataset(mode, file_list=files_list, stage=i, aug=stage_aug))
                logger.info(f'Stage {i} train set size: {len(files_list)}')

    elif mode == 'label_gen':
        for i in range(1, stage + 1):
            file_path = os.path.join(path, f'stage_{i}.txt')
            with open(file_path) as fl:
                files_list = [line.rstrip() for line in fl]
            datasets.append(PatchDataset(mode, file_list=files_list, stage=i, aug=False))

    else:  # test / predict
        files_list = glob.glob(os.path.join(path, '*.zarr'))
        if not files_list:
            raise FileNotFoundError(f'No .zarr patches found in {path}')
        datasets.append(PatchDataset(mode, file_list=files_list, stage=stage, aug=False))

    concat_dataset = ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=WORKERS,
    )
    return dataloader


class PatchDataset(Dataset):
    """
    Loads Landsat 8 patches from Zarr directories.

    Returns a (input_tensor, label, filename) tuple where:
        input_tensor : (17, H+2, W+2) float32
        label        : (1,  H+2, W+2) int64  — 0/1 binary, 255=nodata (ignored)
    """

    def __init__(self, mode, file_list, stage=0, aug=False):
        self.mode      = mode
        self.stage     = stage
        self.file_list = file_list
        self.size      = len(self.file_list)
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transforms = None
        if mode == 'train' and aug:
            self.transforms = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                ZoomIn(),
                Rotate90(),
                CutOut(),
            ])
            logger.info(f'Stage {stage} augmentation enabled')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        patch_path = self.file_list[idx]
        store = zarr.open_group(patch_path, mode='r')

        # ── Spectral bands: normalise DN → TOA reflectance ─────────────
        spectral = store['spectral'][:].astype(np.float32) / 10000.0  # (H, W, 8)

        # ── Precomputed derived features ────────────────────────────────
        rgb   = store['rgb'][:]    # (H, W, 3)
        hsv   = store['hsv'][:]    # (H, W, 3)
        sobel = store['sobel'][:]  # (H, W, 3)

        # Full input: (H, W, 17) = spectral(8) + rgb(3) + hsv(3) + sobel(3)
        full_input = np.concatenate([spectral, rgb, hsv, sobel], axis=-1)

        # ── Labels ──────────────────────────────────────────────────────
        if self.mode == 'train':
            if self.stage == 0:
                label = store['label'][:]
            else:
                if 'pseudo_label' not in store:
                    raise RuntimeError(
                        f'pseudo_label not found in {patch_path} for stage '
                        f'{self.stage}. Run label_generation.py first.')
                label = store['pseudo_label'][:]
        else:
            label = store['label'][:]

        label = label[:, :, None]  # (H, W, 1) = (256, 256, 1)

        # ── Pad label to 258×258 with NODATA before transforms ──────────
        # full_input is already (H+2, W+2, 17) from zarr; label needs to
        # match spatially so flips/rotations are applied consistently.
        p2d   = ((1, 1), (1, 1), (0, 0))
        label = np.pad(label, p2d, 'constant', constant_values=NODATA_LABEL)

        # ── Augmentation (both 258×258) ──────────────────────────────────
        if self.mode == 'train' and self.transforms is not None:
            full_input, label = self.transforms([full_input, label])

        # ── HWC → CHW ────────────────────────────────────────────────────
        full_input = np.transpose(full_input, (2, 0, 1))          # (17, H+2, W+2)
        label      = np.transpose(label, (2, 0, 1)).astype(np.int64)  # (1, H+2, W+2)

        return full_input, label, patch_path
