"""
QA_PIXEL bit-mask mapping for Landsat 8 Collection 2 Level-1.

Converts the QA_PIXEL band into either a 6-class label map or a 3-class
cloud / cloud shadow / no-cloud map.

6-class scheme (class constants below):
    0 – No-Data (Fill)
    1 – Clear-Sky Land
    2 – Cloud
    3 – Cloud Shadow
    4 – Snow / Ice
    5 – Water

3-class scheme:
    0   – No-Cloud        (clear, snow, water)
    1   – Cloud           (cloud, cirrus, dilated cloud)
    2   – Cloud Shadow
    255 – No-Data         (fill)

Bit layout of QA_PIXEL (Landsat 8 Collection 2):
    Bit  0 : Fill
    Bit  1 : Dilated Cloud
    Bit  2 : Cirrus (high confidence)
    Bit  3 : Cloud
    Bit  4 : Cloud Shadow
    Bit  5 : Snow
    Bit  6 : Clear
    Bit  7 : Water
    Bits 8-9 : Cloud Confidence (0=none, 1=low, 2=medium, 3=high)
    Bits 10-11: Cloud Shadow Confidence
    Bits 12-13: Snow/Ice Confidence
    Bits 14-15: Cirrus Confidence
"""

import numpy as np


# ── 6-class constants ──────────────────────────────────────────────────
CLASS_NODATA = 0
CLASS_CLEAR = 1
CLASS_CLOUD = 2
CLASS_SHADOW = 3
CLASS_SNOW = 4
CLASS_WATER = 5

CLASS_NAMES = {
    CLASS_NODATA: 'No-Data',
    CLASS_CLEAR:  'Clear-Sky',
    CLASS_CLOUD:  'Cloud',
    CLASS_SHADOW: 'Shadow',
    CLASS_SNOW:   'Snow/Ice',
    CLASS_WATER:  'Water',
}

NUM_CLASSES = 6

# ── 3-class constants ──────────────────────────────────────────────────
BINARY_NOCLOUD = 0
BINARY_CLOUD   = 1
BINARY_SHADOW  = 2
BINARY_NODATA  = 255

BINARY_CLASS_NAMES = {
    BINARY_NOCLOUD: 'No-Cloud',
    BINARY_CLOUD:   'Cloud',
    BINARY_SHADOW:  'Cloud Shadow',
    BINARY_NODATA:  'No-Data',
}

NUM_BINARY_CLASSES = 3


def qa_pixel_to_classes(qa_pixel: np.ndarray) -> np.ndarray:
    """
    Convert Landsat 8 Collection 2 QA_PIXEL bitmask to the 6-class label map.

    Priority order (highest → lowest):
        Fill → Cloud → Shadow → Snow/Ice → Water → Clear

    Parameters
    ----------
    qa_pixel : np.ndarray
        2-D array of QA_PIXEL values (uint16).

    Returns
    -------
    labels : np.ndarray
        2-D array of class labels (uint8), values 0–5.
    """
    labels = np.full(qa_pixel.shape, CLASS_CLEAR, dtype=np.uint8)

    # Apply in reverse priority so that higher-priority classes overwrite
    # Water (Bit 7)
    labels[_bit_set(qa_pixel, 7)] = CLASS_WATER

    # Snow/Ice (Bit 5)
    labels[_bit_set(qa_pixel, 5)] = CLASS_SNOW

    # Cloud Shadow (Bit 4)
    labels[_bit_set(qa_pixel, 4)] = CLASS_SHADOW

    # Cloud (Bit 3), Cirrus (Bit 2), or Dilated Cloud (Bit 1)
    cloud_mask = _bit_set(qa_pixel, 3) | _bit_set(qa_pixel, 2) | _bit_set(qa_pixel, 1)
    labels[cloud_mask] = CLASS_CLOUD

    # Fill / No-Data (Bit 0)
    labels[_bit_set(qa_pixel, 0)] = CLASS_NODATA

    return labels


def qa_pixel_to_binary(qa_pixel: np.ndarray) -> np.ndarray:
    """
    Convert Landsat 8 Collection 2 QA_PIXEL bitmask to a 3-class label.

    No-Cloud     = Clear (Bit 6) | Snow (Bit 5) | Water (Bit 7)
    Cloud        = Cloud (Bit 3) | Cirrus (Bit 2) | Dilated Cloud (Bit 1)
    Cloud Shadow = Cloud Shadow (Bit 4)   [overridden by Cloud if both set]
    No-Data      = Fill (Bit 0)  → label 255 (ignored during training)

    Parameters
    ----------
    qa_pixel : np.ndarray
        2-D array of QA_PIXEL values (uint16).

    Returns
    -------
    labels : np.ndarray
        2-D array of 3-class labels (uint8):
        0 = no-cloud, 1 = cloud, 2 = cloud shadow, 255 = no-data.
    """
    labels = np.full(qa_pixel.shape, BINARY_NOCLOUD, dtype=np.uint8)

    # Cloud Shadow (applied first, lower priority)
    labels[_bit_set(qa_pixel, 4)] = BINARY_SHADOW

    # Cloud overrides shadow when both bits are set
    cloud_mask = (
        _bit_set(qa_pixel, 1) |  # Dilated Cloud
        _bit_set(qa_pixel, 2) |  # Cirrus
        _bit_set(qa_pixel, 3)    # Cloud
    )
    labels[cloud_mask] = BINARY_CLOUD

    # Fill / No-Data overrides everything
    labels[_bit_set(qa_pixel, 0)] = BINARY_NODATA

    return labels


def _bit_set(arr: np.ndarray, bit: int) -> np.ndarray:
    """Check whether a specific bit is set."""
    return (arr & (1 << bit)) != 0


def get_class_colors():
    """Return a color map (RGB 0-255) for the 6 classes, useful for visualization."""
    return {
        CLASS_NODATA: (0, 0, 0),        # Black
        CLASS_CLEAR:  (34, 139, 34),     # Forest green
        CLASS_CLOUD:  (255, 255, 255),   # White
        CLASS_SHADOW: (128, 128, 128),   # Grey
        CLASS_SNOW:   (0, 255, 255),     # Cyan
        CLASS_WATER:  (0, 0, 255),       # Blue
    }
