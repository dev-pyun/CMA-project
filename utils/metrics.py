"""
Evaluation metrics for semantic segmentation.

Provides mIoU, per-class IoU, Overall Accuracy, F1-score, and
confusion-matrix utilities.  Mirrors the metric interface used
in the original deep-fmask project.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

NUM_CLASSES = 6
CLASS_NAMES = ['nocloud', 'cloud', 'shadow']


class Metrics:
    """Accumulates training / validation metrics across batches and epochs."""

    def __init__(self, device, num_classes=NUM_CLASSES):
        self.device = device
        self.num_classes = num_classes
        self.reset_metrics()

    # ------------------------------------------------------------------
    # Per-step accumulation
    # ------------------------------------------------------------------
    def add_step_info(self, mode, loss, acc=0):
        if mode == 'train':
            self.train_loss_list.append(loss.item())
            self.train_acc_list.append(acc.item() if torch.is_tensor(acc) else acc)
        else:
            self.val_loss_list.append(loss.item())

    # ------------------------------------------------------------------
    # Epoch-level aggregation
    # ------------------------------------------------------------------
    def aggregate_metrics(self, epoch):
        train_loss = np.mean(self.train_loss_list) if self.train_loss_list else 0
        train_acc = np.mean(self.train_acc_list) if self.train_acc_list else 0
        val_loss = np.mean(self.val_loss_list) if self.val_loss_list else 0

        # Per-class IoU from confusion matrix
        cm = self.val_confusion_matrix.cpu().numpy()
        class_iou = {}
        for c in range(self.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            class_iou[name] = tp / denom if denom > 0 else 0.0

        miou = np.mean(list(class_iou.values()))
        self.val_mIoU_history.append(miou)

        overall_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

        train_metrics = {'loss': train_loss, 'acc': train_acc}
        valid_metrics = {'loss': val_loss, 'mIoU': miou, 'OA': overall_acc}

        per_class_str = '  '.join(
            f'{name}: {iou:.4f}' for name, iou in class_iou.items())
        logger.info(
            f'Epoch {epoch + 1}  '
            f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  '
            f'Val Loss: {val_loss:.4f}  mIoU: {miou:.4f}  OA: {overall_acc:.4f}  '
            f'[{per_class_str}]'
        )

        return train_metrics, valid_metrics, class_iou

    # ------------------------------------------------------------------
    # Reset between epochs
    # ------------------------------------------------------------------
    def reset_metrics(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long,
            device=self.device)
        if not hasattr(self, 'val_mIoU_history'):
            self.val_mIoU_history = []


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

def calculate_confusion_matrix(predicted, labels, mode=None,
                               num_classes=NUM_CLASSES):
    """
    Compute the confusion matrix for a batch.

    Parameters
    ----------
    predicted : torch.Tensor  (B, H, W)
    labels    : torch.Tensor  (B, H, W)

    Returns
    -------
    cm : torch.Tensor  (num_classes, num_classes)
    """
    mask = (labels >= 0) & (labels < num_classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long,
                     device=predicted.device)
    idx = num_classes * labels[mask].long() + predicted[mask].long()
    cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(
        num_classes, num_classes)
    return cm


def calculate_accuracy(predicted, labels, mode=None):
    """Pixel-level accuracy for a batch, excluding nodata pixels (label == 255)."""
    mask = (labels >= 0) & (labels < 255)
    correct = (predicted[mask] == labels[mask]).float().sum()
    total   = mask.float().sum()
    return correct / total if total > 0 else torch.tensor(0.0)
