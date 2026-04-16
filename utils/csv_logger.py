"""
CSV logging utilities for tracking training and validation statistics.
"""

import csv
import os
import logging

logger = logging.getLogger(__name__)


def make_overall_statistics_csv(train_metrics, valid_metrics,
                                class_metrics_dict, epoch, log_path):
    """
    Append one row of training/validation statistics to the CSV log.

    Creates the file with a header if it does not exist yet.
    """
    csv_path = os.path.join(log_path, 'training_log.csv')
    file_exists = os.path.isfile(csv_path)

    fieldnames = ['epoch',
                  'train_loss', 'train_acc',
                  'val_loss', 'val_mIoU', 'val_OA']
    # Add per-class IoU fields
    for c in sorted(class_metrics_dict.keys()):
        fieldnames.append(f'class_{c}_IoU')

    row = {
        'epoch': epoch + 1,
        'train_loss': f"{train_metrics['loss']:.6f}",
        'train_acc': f"{train_metrics['acc']:.6f}",
        'val_loss': f"{valid_metrics['loss']:.6f}",
        'val_mIoU': f"{valid_metrics['mIoU']:.6f}",
        'val_OA': f"{valid_metrics['OA']:.6f}",
    }
    for c, iou in class_metrics_dict.items():
        row[f'class_{c}_IoU'] = f'{iou:.6f}'

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_val_csv_metrics(best_epoch, log_path):
    """Print validation metrics for the best epoch."""
    csv_path = os.path.join(log_path, 'training_log.csv')
    if not os.path.isfile(csv_path):
        logger.warning('Training log CSV not found.')
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['epoch']) == best_epoch:
                logger.info(f'Best epoch {best_epoch} metrics:')
                for k, v in row.items():
                    logger.info(f'  {k}: {v}')
                return
    logger.warning(f'Epoch {best_epoch} not found in log.')
