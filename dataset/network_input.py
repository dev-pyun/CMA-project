"""
Flexible band selection and spectral index computation for Landsat 8.

Supports:
  1. Preset modes (swirndsi, swir, vnir, rgb, all, etc.)
  2. Custom mode — select any combination of bands + derived indices via CLI

Landsat 8 band layout in HDF5 (channel index):
    0: B1  Coastal/Aerosol  (0.43–0.45 µm)
    1: B2  Blue             (0.45–0.51 µm)
    2: B3  Green            (0.53–0.60 µm)
    3: B4  Red              (0.63–0.68 µm)
    4: B5  NIR              (0.85–0.88 µm)
    5: B6  SWIR1            (1.57–1.65 µm)
    6: B7  SWIR2            (2.11–2.29 µm)
    7: B9  Cirrus           (1.36–1.38 µm)  — only if include_cirrus=True

Note: indices 0–6 are always present; index 7 (Cirrus) is optional.
"""

import torch

# ======================================================================
# Band name → H5 channel index mapping
# ======================================================================
BAND_INDEX = {
    'B1': 0, 'Coastal': 0,
    'B2': 1, 'Blue': 1,
    'B3': 2, 'Green': 2,
    'B4': 3, 'Red': 3,
    'B5': 4, 'NIR': 4,
    'B6': 5, 'SWIR1': 5,
    'B7': 6, 'SWIR2': 6,
    'B9': 7, 'Cirrus': 7,
}

# ======================================================================
# Spectral index computation functions
# All operate on (B, C, H, W) tensors and return (B, 1, H, W)
# ======================================================================

def _safe_ratio(num, den):
    """Compute num/den, handling division by zero."""
    den = den.clone()
    den[den == 0] = 1
    return num / den


def compute_NDSI(inp_img):
    """Normalized Difference Snow Index = (Green - SWIR1) / (Green + SWIR1)"""
    green = inp_img[:, BAND_INDEX['Green'], :, :]
    swir1 = inp_img[:, BAND_INDEX['SWIR1'], :, :]
    return _safe_ratio(green - swir1, green + swir1).unsqueeze(1)


def compute_NDWI(inp_img):
    """Normalized Difference Water Index = (Green - NIR) / (Green + NIR)"""
    green = inp_img[:, BAND_INDEX['Green'], :, :]
    nir = inp_img[:, BAND_INDEX['NIR'], :, :]
    return _safe_ratio(green - nir, green + nir).unsqueeze(1)


def compute_NDVI(inp_img):
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)"""
    nir = inp_img[:, BAND_INDEX['NIR'], :, :]
    red = inp_img[:, BAND_INDEX['Red'], :, :]
    return _safe_ratio(nir - red, nir + red).unsqueeze(1)


def compute_MNDWI(inp_img):
    """Modified NDWI = (Green - SWIR1) / (Green + SWIR1)"""
    green = inp_img[:, BAND_INDEX['Green'], :, :]
    swir1 = inp_img[:, BAND_INDEX['SWIR1'], :, :]
    return _safe_ratio(green - swir1, green + swir1).unsqueeze(1)


def compute_BSI(inp_img):
    """Bare Soil Index = ((SWIR1+Red) - (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))"""
    swir1 = inp_img[:, BAND_INDEX['SWIR1'], :, :]
    red = inp_img[:, BAND_INDEX['Red'], :, :]
    nir = inp_img[:, BAND_INDEX['NIR'], :, :]
    blue = inp_img[:, BAND_INDEX['Blue'], :, :]
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue)
    return _safe_ratio(num, den).unsqueeze(1)


# Registry of available spectral indices
INDEX_REGISTRY = {
    'NDSI': compute_NDSI,
    'NDWI': compute_NDWI,
    'NDVI': compute_NDVI,
    'MNDWI': compute_MNDWI,
    'BSI': compute_BSI,
}


# ======================================================================
# Preset input mode functions
# ======================================================================

def inp_all(inp_img):
    """All 7 core bands: B1–B7."""
    return inp_img[:, 0:7, :, :]


def inp_all_cirrus(inp_img):
    """All 8 bands including Cirrus: B1–B7 + B9."""
    return inp_img[:, 0:8, :, :]


def inp_rgb(inp_img):
    """RGB: B2, B3, B4 (3 channels)."""
    return inp_img[:, (1, 2, 3), :, :]


def inp_vnir(inp_img):
    """Visible + NIR: B2, B3, B4, B5 (4 channels)."""
    return inp_img[:, (1, 2, 3, 4), :, :]


def inp_swir(inp_img):
    """SWIR: B2, B3, B4, B5, B6, B7 (6 channels)."""
    return inp_img[:, (1, 2, 3, 4, 5, 6), :, :]


def inp_swirndsi(inp_img):
    """SWIR + NDSI: B2, B3, B4, B5, B6, B7 + NDSI (7 channels).
    Default mode matching the paper's swirndsi configuration."""
    bands = inp_img[:, (1, 2, 3, 4, 5, 6), :, :]
    ndsi = compute_NDSI(inp_img)
    return torch.cat([bands, ndsi], dim=1)


def inp_swirndsi_ndwi(inp_img):
    """SWIR + NDSI + NDWI: B2, B3, B4, B5, B6, B7 + NDSI + NDWI (8 channels)."""
    bands = inp_img[:, (1, 2, 3, 4, 5, 6), :, :]
    ndsi = compute_NDSI(inp_img)
    ndwi = compute_NDWI(inp_img)
    return torch.cat([bands, ndsi, ndwi], dim=1)


def inp_swirndwi(inp_img):
    """SWIR + NDWI: B2, B3, B4, B5, B6, B7 + NDWI (7 channels)."""
    bands = inp_img[:, (1, 2, 3, 4, 5, 6), :, :]
    ndwi = compute_NDWI(inp_img)
    return torch.cat([bands, ndwi], dim=1)


def inp_allndsi(inp_img):
    """All bands + NDSI: B1–B7 + NDSI (8 channels)."""
    bands = inp_img[:, 0:7, :, :]
    ndsi = compute_NDSI(inp_img)
    return torch.cat([bands, ndsi], dim=1)


def inp_cirrus_ndsi(inp_img):
    """Cirrus + SWIR + NDSI: B2, B3, B4, B5, B6, B7, B9 + NDSI (8 channels)."""
    bands = inp_img[:, (1, 2, 3, 4, 5, 6, 7), :, :]
    ndsi = compute_NDSI(inp_img)
    return torch.cat([bands, ndsi], dim=1)


# ======================================================================
# Preset registry
# ======================================================================

_PRESET_MODES = {
    'all':           (inp_all, 7),
    'all_cirrus':    (inp_all_cirrus, 8),
    'rgb':           (inp_rgb, 3),
    'vnir':          (inp_vnir, 4),
    'swir':          (inp_swir, 6),
    'swirndsi':      (inp_swirndsi, 7),
    'swirndsindwi':  (inp_swirndsi_ndwi, 8),
    'swirndwi':      (inp_swirndwi, 7),
    'allndsi':       (inp_allndsi, 8),
    'cirrus_ndsi':   (inp_cirrus_ndsi, 8),
}


# ======================================================================
# Custom mode — build input function from band names / index names
# ======================================================================

class CustomInput:
    """
    Dynamically builds an input selection function from user-specified
    bands and indices.

    Usage:
        custom = CustomInput(bands=['B2','B3','B4','B5','B6','B7'],
                              indices=['NDSI', 'NDWI'])
        output = custom(inp_img)   # → (B, 8, H, W)
    """

    def __init__(self, bands=None, indices=None):
        self.band_indices = []
        self.band_names = bands or []
        self.index_names = indices or []

        for b in self.band_names:
            if b in BAND_INDEX:
                self.band_indices.append(BAND_INDEX[b])
            else:
                raise ValueError(
                    f'Unknown band: {b}. '
                    f'Available: {list(BAND_INDEX.keys())}')

        self.index_fns = []
        for idx_name in self.index_names:
            if idx_name in INDEX_REGISTRY:
                self.index_fns.append(INDEX_REGISTRY[idx_name])
            else:
                raise ValueError(
                    f'Unknown index: {idx_name}. '
                    f'Available: {list(INDEX_REGISTRY.keys())}')

        self.n_channels = len(self.band_indices) + len(self.index_fns)
        self.__name__ = f'custom_{"_".join(self.band_names)}_{"_".join(self.index_names)}'

    def __call__(self, inp_img):
        parts = []
        if self.band_indices:
            parts.append(inp_img[:, self.band_indices, :, :])
        for fn in self.index_fns:
            parts.append(fn(inp_img))
        return torch.cat(parts, dim=1)


# ======================================================================
# Public API
# ======================================================================

def get_inp_func(mode, bands=None, indices=None):
    """
    Get the input transformation function.

    Parameters
    ----------
    mode : str
        Preset mode name (e.g. 'swirndsi') or 'custom'.
    bands : list of str, optional
        Band names for custom mode (e.g. ['B2', 'B3', 'B4', 'B5']).
    indices : list of str, optional
        Index names for custom mode (e.g. ['NDSI', 'NDWI']).

    Returns
    -------
    callable
        Function that transforms (B, C_all, H, W) → (B, C_selected, H, W).
    """
    if mode == 'custom':
        if not bands and not indices:
            raise ValueError(
                "Custom mode requires at least one of --bands or --indices")
        return CustomInput(bands=bands, indices=indices)

    if mode in _PRESET_MODES:
        return _PRESET_MODES[mode][0]

    raise ValueError(
        f'Unknown input mode: {mode}. '
        f'Available presets: {list(_PRESET_MODES.keys())} or use "custom"')


def get_inp_channels(mode, bands=None, indices=None):
    """
    Get the number of input channels for a given mode.

    Parameters
    ----------
    mode : str
        Preset mode name or 'custom'.
    bands : list of str, optional
        Band names for custom mode.
    indices : list of str, optional
        Index names for custom mode.

    Returns
    -------
    int
        Number of input channels.
    """
    if mode == 'custom':
        n_bands = len(bands) if bands else 0
        n_indices = len(indices) if indices else 0
        return n_bands + n_indices

    if mode in _PRESET_MODES:
        return _PRESET_MODES[mode][1]

    raise ValueError(f'Unknown input mode: {mode}')


def list_available_modes():
    """Print all available input modes and their channel counts."""
    print('\n=== Preset Input Modes ===')
    for name, (fn, n_ch) in _PRESET_MODES.items():
        print(f'  {name:20s}  →  {n_ch} channels')

    print('\n=== Available Spectral Indices ===')
    for name in INDEX_REGISTRY:
        print(f'  {name}')

    print('\n=== Available Bands ===')
    seen = set()
    for name, idx in BAND_INDEX.items():
        if idx not in seen:
            print(f'  {name} (index {idx})')
            seen.add(idx)
