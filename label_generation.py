"""
Pseudo-label generation for the self-training pipeline.

Uses a trained model from stage N to generate improved labels
for the training data that will be used in stage N+1.

Usage:
    python label_generation.py -e exp1_stage0 -st 1 -gpu 0 1
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.dir_paths import TRAIN_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('label_generation')


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Pseudo-Label Generation')
    parser.add_argument('-e', '--exp_name', required=True,
                        help='Name of the trained experiment')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-st', '--stage', type=int, default=1,
                        help='Target stage for label generation')
    parser.add_argument('-ep', '--model_epoch', type=int, default=0,
                        help='Model epoch to load (0 = best model)')
    parser.add_argument('-ip', '--inp_mode', default='swirndsi',
                        help='Input mode (must match training)')
    parser.add_argument('--bands', nargs='+', default=None,
                        help='Band names for custom mode')
    parser.add_argument('--indices', nargs='+', default=None,
                        help='Index names for custom mode')
    parser.add_argument('-gpu', '--gpu_id', type=int, nargs='+', default=[0],
                        help='GPU IDs')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args, mode='label_gen')

    # Set up dataloader for label generation
    test_loader = setup_data(
        args.batch_size, 'label_gen',
        stage=args.stage,
        path=TRAIN_PATH)

    # Initialize model for evaluation
    model = Model(exp, gpu_id=args.gpu_id)
    model.network.eval()

    loader_itr = tqdm(
        test_loader,
        total=len(test_loader),
        leave=False,
        desc='Pseudo-Label Generation')

    # Generate pseudo-labels
    import torch
    with torch.no_grad():
        for batch, network_input in enumerate(loader_itr):
            model.valid_step(network_input, mode='label_gen')

    # Save per-class frequency statistics
    model.write_stage_stats()
    logger.info('Pseudo-label generation complete.')
