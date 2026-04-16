"""
Data augmentation transforms for semantic segmentation.

Each transform operates on a list [spectral_image, labels] where both are
numpy arrays in (H, W, C) format.  Labels are transformed with nearest
interpolation to preserve class values.
"""

import random

import numpy as np
from cv2 import resize, INTER_CUBIC, INTER_NEAREST


class HorizontalFlip:
    """Randomly flip horizontally with 50% probability."""

    def __call__(self, data):
        if random.random() > 0.5:
            data = [np.flip(d, axis=1).copy() for d in data]
        return data


class VerticalFlip:
    """Randomly flip vertically with 50% probability."""

    def __call__(self, data):
        if random.random() > 0.5:
            data = [np.flip(d, axis=0).copy() for d in data]
        return data


class Rotate90:
    """Randomly rotate by 90° increments with 50% probability."""

    def __call__(self, data):
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            data = [np.rot90(d, k, axes=(0, 1)).copy() for d in data]
        return data


class CutOut:
    """
    Randomly zero out a rectangular patch (CutOut augmentation).
    Patch size is 10–30% of image size.
    """

    def __call__(self, data):
        if random.random() > 0.5:
            h, w = data[0].shape[:2]
            cut_h = random.randint(int(h * 0.1), int(h * 0.3))
            cut_w = random.randint(int(w * 0.1), int(w * 0.3))
            y = random.randint(0, h - cut_h)
            x = random.randint(0, w - cut_w)
            data[0][y:y + cut_h, x:x + cut_w, :] = 0
            data[1][y:y + cut_h, x:x + cut_w, :] = 0
        return data


class ZoomIn:
    """
    Randomly zoom into a sub-region and resize back to original size.
    Zoom factor: 1.0–1.5×.
    """

    def __call__(self, data):
        if random.random() > 0.5:
            h, w = data[0].shape[:2]
            zoom = random.uniform(1.0, 1.5)
            new_h = int(h / zoom)
            new_w = int(w / zoom)
            y = random.randint(0, h - new_h)
            x = random.randint(0, w - new_w)

            cropped_img = data[0][y:y + new_h, x:x + new_w, :]
            cropped_lbl = data[1][y:y + new_h, x:x + new_w, :]

            data[0] = resize(cropped_img, (w, h),
                             interpolation=INTER_CUBIC)
            data[1] = resize(cropped_lbl, (w, h),
                             interpolation=INTER_NEAREST)
            if data[1].ndim == 2:
                data[1] = data[1][:, :, None]
        return data
