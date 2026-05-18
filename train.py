"""
Training script for each stage of the self-training pipeline.

Supports:
  - Supervised mode (--full): train largest network on all data with Fmask labels
  - Self-training mode: train progressively larger networks at each stage
  - Preset and custom band/index input modes
  - Multi-GPU training (DataParallel)

Usage examples:
    # Stage 0, default swirndsi mode
    python train.py -e exp1_stage0 -st 0 -gpu 0 1

    # Full supervised mode
    python train.py -e exp1_full --full -gpu 0 1

    # Custom bands + indices
    python train.py -e exp1_stage0 -st 0 --inp_mode custom \
        --bands B2 B3 B4 B5 B6 --indices NDSI NDWI
"""

import argparse
import logging
import sys
import os
import torch

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

from dataset.patch_dataset import setup_data, set_seed
from network.model import Model
from utils.MFB import get_MFB_weights
from utils.dir_paths import TRAIN_PATH, VALID_PATH
from utils.experiment import Experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('train_script')


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('-e', '--exp_name', required=True,
                        help='Name of experiment')

    # Network configuration
    parser.add_argument('--full', dest='full', action='store_true',
                        default=False,
                        help='Train the largest network using all data '
                             'with QA_PIXEL labels (supervised mode)')
    parser.add_argument('-st', '--stage', type=int, default=0,
                        help='Self-training pipeline stage (0–3)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6,
                        help='Learning rate (default: 1e-6)')
    parser.add_argument('--no_dropout', dest='dropout',
                        action='store_false', default=True,
                        help='Disable dropout')
    parser.add_argument('-ep', '--num_epochs', type=int, default=400,
                        help='Maximum training epochs (default: 400)')

    # Data
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--no_aug', dest='aug',
                        action='store_false', default=True,
                        help='Disable data augmentation')
    parser.add_argument('--reset_stage_data', dest='reset_stage_data',
                        action='store_true', default=False,
                        help='Reassign H5 files for each stage')

    # Input mode — preset or custom
    parser.add_argument('-ip', '--inp_mode', default='swirndsi',
                        help='Input mode: preset name or "custom". '
                             'Presets: swirndsi, all, rgb, vnir, swir, '
                             'allndsi, swirndwi, swirndsindwi, cirrus_ndsi')
    parser.add_argument('--bands', nargs='+', default=None,
                        help='Band names for custom mode '
                             '(e.g. --bands B2 B3 B4 B5 B6)')
    parser.add_argument('--indices', nargs='+', default=None,
                        help='Index names for custom mode '
                             '(e.g. --indices NDSI NDWI NDVI)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Hardware
    parser.add_argument('-gpu', '--gpu_id', type=int, nargs='+', default=[0],
                        help='GPU IDs (e.g. -gpu 0 1)')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()

    # Store custom band/index info in args for Experiment
    if args.inp_mode == 'custom':
        from dataset.network_input import CustomInput
        # Validate early
        _ = CustomInput(bands=args.bands, indices=args.indices)

    exp = Experiment(args)
    set_seed(args.seed)

    # Set up data loaders
    train_loader = setup_data(
        args.batch_size, mode='train',
        stage=args.stage, path=TRAIN_PATH,
        full=args.full, aug=args.aug,
        reset=args.reset_stage_data)

    test_loader = setup_data(args.batch_size, mode='test', path=VALID_PATH)

    # Initialize model
    model = Model(exp, gpu_id=args.gpu_id)

    # Compute class weights via Median Frequency Balancing
    exp.weights = get_MFB_weights(train_loader)

    # ---- Training loop ----
    for epoch in range(args.num_epochs):
        logger.info(f'Epoch {epoch + 1}/{args.num_epochs}')

        # Train
        model.network.train()
        loader_itr_train = tqdm(
            train_loader,
            total=len(train_loader),
            leave=False,
            desc=f'Train Epoch {epoch + 1}')

        for batch, network_input in enumerate(loader_itr_train):
            model.train_step(network_input)

        # Validate
        model.network.eval()
        with torch.no_grad():
            loader_itr = tqdm(
                test_loader,
                total=len(test_loader),
                leave=False,
                desc=f'Valid Epoch {epoch + 1}')

            for batch, network_input in enumerate(loader_itr):
                model.valid_step(network_input)

        # Check early stopping
        early_stop_flag = model.refresh_stats()
        if early_stop_flag:
            break

    model.save_best_model()
    logger.info('Training complete.')
