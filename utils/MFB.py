"""
Median Frequency Balancing (MFB) for computing class weights.

The weight for each class is:
    w_c = median(freq) / freq_c

where freq_c is the relative frequency of class c across the training set.
This gives lower weight to common classes and higher weight to rare ones.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

NUM_CLASSES = 3  # 3-class: 0=no-cloud, 1=cloud, 2=shadow


def get_MFB_weights(dataloader, num_classes=NUM_CLASSES, max_weight=10.0):
    """
    Iterate over the dataloader once and compute MFB weights.

    Parameters
    ----------
    max_weight : float
        Upper bound for any single class weight. Prevents weight explosion
        when a class has zero or near-zero frequency (e.g. shadow absent
        from QA_PIXEL labels in polar scenes).

    Returns
    -------
    weights : np.ndarray  shape (num_classes,)
    """
    total_freq = np.zeros(num_classes, dtype=np.float64)

    for _, labels, _ in dataloader:
        lbl = labels[:, 0, :, :].numpy()  # (B, H, W)
        for c in range(num_classes):
            total_freq[c] += (lbl == c).sum()  # nodata(255) excluded since 255 >= num_classes

    total_pixels = total_freq.sum()
    if total_pixels == 0:
        logger.warning('No pixels found – returning uniform weights.')
        return np.ones(num_classes, dtype=np.float32)

    freq = total_freq / total_pixels
    freq[freq == 0] = 1e-10  # avoid division by zero
    median_freq = np.median(freq[freq > 1e-10])

    weights = np.clip(median_freq / freq, 0, max_weight)
    logger.info(f'Class frequencies: {freq}')
    logger.info(f'MFB weights:       {weights}')
    return weights.astype(np.float32)


def calculate_file_freq(label_np, num_classes=NUM_CLASSES):
    """
    Compute per-class pixel frequencies for a single label patch.
    Nodata pixels (label == 255) are excluded from the denominator.

    Parameters
    ----------
    label_np : np.ndarray  (H, W)

    Returns
    -------
    freq : np.ndarray  shape (num_classes,)
    """
    freq  = np.zeros(num_classes, dtype=np.float64)
    valid = label_np[label_np < num_classes]  # exclude nodata (255)
    total = valid.size
    if total == 0:
        return freq
    for c in range(num_classes):
        freq[c] = (valid == c).sum() / total
    return freq
