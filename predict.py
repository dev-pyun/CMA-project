"""
Prediction script — run a trained model on new Landsat 8 scenes.

Usage:
    python predict.py -e exp1_stage3 -p /path/to/test/scenes/ -gpu 0 1
"""

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.metrics import Metrics, calculate_confusion_matrix
from utils.qa_pixel_mapping import CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('predict')


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Prediction Script')
    parser.add_argument('-e', '--exp_name', required=True,
                        help='Experiment name (e.g. exp1_stage3)')
    parser.add_argument('-p', '--path', required=True,
                        help='Path to folder with H5 test patches')
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='Batch size')
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

    # Hack: set stage and other training args for Experiment compatibility
    args.stage = 3
    args.full = False
    args.dropout = True
    args.learning_rate = 1e-6

    exp = Experiment(args, mode='test')

    # Load test data
    test_loader = setup_data(
        batch_size=args.batch_size,
        mode='test',
        path=os.path.abspath(args.path))

    # Initialize model
    model = Model(exp, gpu_id=args.gpu_id)
    model.network.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = Metrics(device)

    loader_itr = tqdm(
        test_loader,
        total=len(test_loader),
        leave=False,
        desc='Predicting')

    with torch.no_grad():
        for batch, network_input in enumerate(loader_itr):
            model.valid_step(network_input, mode='test')

    # Print results
    cm = model.metrics.val_confusion_matrix.cpu().numpy()

    print('\n' + '=' * 60)
    print('PREDICTION RESULTS')
    print('=' * 60)

    # Per-class IoU
    print('\nPer-class IoU:')
    ious = []
    for c in range(6):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)
        print(f'  {CLASS_NAMES[c]:12s}: {iou:.4f}')

    miou = np.mean(ious)
    oa = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
    print(f'\nmIoU: {miou:.4f}')
    print(f'Overall Accuracy: {oa:.4f}')

    # Confusion matrix
    print('\nConfusion Matrix:')
    header = '            ' + ''.join(f'{CLASS_NAMES[c]:>10s}' for c in range(6))
    print(header)
    for c in range(6):
        row = f'{CLASS_NAMES[c]:12s}' + ''.join(f'{cm[c, j]:10d}' for j in range(6))
        print(row)
