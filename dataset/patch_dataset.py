"""
Patch dataset loader for Landsat 8 HDF5 files.

Loads patches created by utils/split_scene.py and serves them as
(spectral_image, labels, filename) tuples for training and validation.

HDF5 data layout:
    channels 0–6 : B1–B7 spectral bands (+ optional B9 at channel 7)
    channel  N   : QA_PIXEL labels (6-class, added by split_scene.py)
    channel  N+2 : pseudo-labels (added by label_generation.py in stage 1+)
"""

import glob
import logging
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

from dataset.transforms import VerticalFlip, HorizontalFlip, Rotate90, CutOut, ZoomIn

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

WORKERS = 16

# Number of core spectral bands (B1–B7)
N_SPECTRAL_BANDS = 7


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
    Assign H5 files to different stages of the self-training pipeline.
    Stage 0 gets stage_0_ratio of the data, the rest is split evenly
    among stages 1–3.
    """
    h5_folder = os.path.abspath(h5_folder)
    file_list = glob.glob(os.path.join(h5_folder, '*.h5'))
    n_files = len(file_list)
    if not n_files:
        raise FileNotFoundError(f'No h5 files found in {h5_folder}')

    # Full file list
    train_list_filename = os.path.join(h5_folder, 'stage_full.txt')
    with open(train_list_filename, 'w') as f:
        for train_file in file_list:
            f.write(f'{train_file}\n')

    unlabelled_ratio = 1 - stage_0_ratio
    unlabelled_size = int(n_files * unlabelled_ratio / (stages - 1))
    labeled_size = n_files - (stages - 1) * unlabelled_size

    random.shuffle(file_list)

    start = 0
    for count in range(stages):
        stage_size = labeled_size if count == 0 else unlabelled_size
        end = min(n_files, start + stage_size)
        stage_list = file_list[start:end]

        stage_filename = os.path.join(h5_folder, f'stage_{count}.txt')
        with open(stage_filename, 'w') as f:
            for train_file in stage_list:
                f.write(f'{train_file}\n')
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
        Path to the H5 directory.
    full : bool
        If True, use all data (supervised mode).
    aug : bool
        Whether to apply data augmentation.
    reset : bool
        Re-generate stage assignment files.
    """
    datasets = []
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        if stage != 0 and reset:
            logger.warning('Stage data reset only in stage 0. '
                           'Setting reset_stage_data to False.')
            reset = False

        if not check_data_split(path, reset=reset):
            split_data(path)

        if full:
            file_path = os.path.join(path, 'stage_full.txt')
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(mode, file_list=files_list, stage=0, aug=aug))
            logger.info(f'Full train set size: {len(files_list)}')
        else:
            for i in range(stage + 1):
                file_path = os.path.join(path, f'stage_{i}.txt')
                with open(file_path, 'r') as fl:
                    files_list = [line.rstrip() for line in fl.readlines()]

                    stage_aug = bool(stage) if aug else aug

                    datasets.append(
                        PatchDataset(mode, file_list=files_list,
                                     stage=i, aug=stage_aug))
                logger.info(f'Stage {i} train set size: {len(files_list)}')

    elif mode == 'label_gen':
        for i in range(1, stage + 1):
            file_path = os.path.join(path, f'stage_{i}.txt')
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(mode, file_list=files_list,
                                 stage=i, aug=False))

    else:  # test / predict
        files_list = glob.glob(os.path.join(path, '*.h5'))
        if not files_list:
            raise FileNotFoundError(f'H5 files not found in {path}')
        datasets.append(
            PatchDataset(mode, file_list=files_list, stage=stage, aug=False))

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
    Loads Landsat 8 patches from HDF5 files.

    Data layout in each H5 file:
        data[:, :, 0:N_BANDS]  → spectral bands
        data[:, :, N_BANDS]    → QA_PIXEL labels (stage 0 training labels)
        data[:, :, N_BANDS+2]  → pseudo-labels (stage 1+ training labels)
    """

    def __init__(self, mode, file_list, stage=0, aug=False):
        self.mode = mode
        self.stage = stage
        self.file_list = file_list
        self.size = len(self.file_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        with h5py.File(self.file_list[idx], 'r') as hf:
            spectral_image = hf.get('data')[:]

        # Split into spectral bands and labels
        labels = spectral_image[:, :, N_SPECTRAL_BANDS:].astype(np.uint8)
        labels[labels < 0] = 0

        spectral_image = spectral_image[:, :, :N_SPECTRAL_BANDS].astype(np.float32)

        # Select which label channel to use based on training stage
        if self.mode == 'train':
            if self.stage == 0:
                # Use QA_PIXEL labels (index 0 in labels array)
                lbl_idx = 0
            else:
                # Use pseudo-labels (index 2 in labels array)
                lbl_idx = 2
                if lbl_idx >= labels.shape[-1]:
                    raise RuntimeError(
                        f'Pseudo-labels not found for stage {self.stage}. '
                        f'Run label_generation.py first.')

            labels = labels[:, :, lbl_idx][:, :, None]

        # Apply augmentation
        if self.mode == 'train' and self.transforms is not None:
            transform_input = [spectral_image, labels]
            transform_out = self.transforms(transform_input)
            spectral_image, labels = transform_out[0], transform_out[1]

        # Padding (1 pixel border)
        p2d = ((1, 1), (1, 1), (0, 0))
        spectral_image = np.pad(spectral_image, p2d, 'constant',
                                constant_values=0)
        labels = np.pad(labels, p2d, 'constant', constant_values=0)

        # Normalize reflectance values (DN to TOA reflectance scale)
        spectral_image = spectral_image / 10000.0

        # Convert HWC → CHW
        spectral_image = np.transpose(spectral_image, (2, 0, 1))
        labels = np.transpose(labels, (2, 0, 1)).astype(np.int64)

        return spectral_image, labels, self.file_list[idx]
