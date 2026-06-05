"""
Validation confusion matrix evaluation for all 20 trained models.

Runs 5 experiment families in parallel (one process each); each process
evaluates 4 stages (stage0→3) sequentially on the 8 GT-labelled scenes.

Output per model (one model = one stage of one experiment):
  val_confusion/{exp_base}_stage{N}/gt_vs_fmask.png
  val_confusion/{exp_base}_stage{N}/gt_vs_{exp_base}.png
  val_confusion/{exp_base}_stage{N}/fmask_vs_{exp_base}.png
  val_confusion/{exp_base}_stage{N}/cm_gt_fmask.npy   (raw CM arrays)
  val_confusion/{exp_base}_stage{N}/cm_gt_model.npy
  val_confusion/{exp_base}_stage{N}/cm_fmask_model.npy
  val_confusion/summary_metrics.csv   (OA + per-class IoU + mIoU)
"""

import csv
import multiprocessing as mp
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.network_input import get_inp_func
from utils.scene_inference import (
    load_model, load_scene_bands, load_cfmask, load_gt_labels,
    run_scene_inference, accumulate_cm,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

PREPARED_DIR = '/home/pyuncb/src/label_code/prepared'
LABELS_DIR   = '/home/pyuncb/src/label_code/labels'
OUT_DIR      = '/home/pyuncb/src/val_confusion'
LOG_DIR      = '/home/pyuncb/src/logs'

# ── Scenes with GT labels ─────────────────────────────────────────────────────

GT_SCENES = [
    'LC08_L1GT_165110_20200302_20201016_02_T2',
    'LC08_L1GT_171110_20200225_20201016_02_T2',
    'LC08_L1GT_177110_20200219_20201016_02_T2',
    'LC08_L1GT_181098_20200419_20201016_02_T2',
    'LC08_L1GT_188114_20201114_20210315_02_T2',
    'LC08_L1GT_199105_20201213_20210314_02_T2',
    'LC08_L1GT_199110_20200128_20201016_02_T2',
    'LC08_L1GT_200111_20201017_20201105_02_T2',
]

# ── Experiment config ─────────────────────────────────────────────────────────

EXPERIMENTS = [
    {'base': 'exp_cirrus_ndsi',     'inp_mode': 'cirrus_ndsi',   'num_classes': 2},
    {'base': 'exp_ndsi679',         'inp_mode': 'ndsi679',        'num_classes': 2},
    {'base': 'exp_swirndsi_pca3',   'inp_mode': 'swirndsi_pca3', 'num_classes': 3},
    {'base': 'swirndsi_trial2',     'inp_mode': 'swirndsi',       'num_classes': 2},
    {'base': 'swirndsindwi_trial1', 'inp_mode': 'swirndsindwi',  'num_classes': 2},
]

CLASS_NAMES = {
    2: ['no-cloud', 'cloud'],
    3: ['no-cloud', 'cloud', 'shadow'],
}

STAGES = [0, 1, 2, 3]


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_confusion_matrix_png(cm: np.ndarray, row_label: str, col_label: str,
                               class_names: list, save_path: str,
                               nodata_row: np.ndarray = None) -> None:
    """
    Save confusion matrix as PNG with per-cell counts and row-normalised %.

    nodata_row : (n_classes,) int64 — optional extra row showing what
                 Fmask/Model predicted for GT-nodata pixels.
                 Displayed at the bottom in grey; excluded from OA/IoU.
    """
    n     = cm.shape[0]
    total = cm.sum()
    oa    = cm.diagonal().sum() / total if total > 0 else 0.0

    has_nodata = nodata_row is not None
    n_rows     = n + (1 if has_nodata else 0)
    row_names  = list(class_names) + (['nodata'] if has_nodata else [])

    # Build display matrix: stack nodata row if present
    display = cm.astype(float)
    if has_nodata:
        display = np.vstack([display, nodata_row.reshape(1, n).astype(float)])

    fig, ax = plt.subplots(figsize=(4 + n * 1.5, 3 + n_rows * 1.5))

    # Colour the valid and nodata regions separately
    vis = display.copy()
    if has_nodata:
        # Normalise colour scale on the valid CM only, then map nodata row to grey
        vis[-1, :] = -1  # sentinel — will paint grey
    cmap = plt.cm.Blues
    cmap.set_under('lightgrey')
    im = ax.imshow(vis, interpolation='nearest', cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_rows):
        row_arr = display[i]
        row_sum = row_arr.sum()
        for j in range(n):
            val = display[i, j]
            pct = val / row_sum * 100 if row_sum > 0 else 0.0
            color = 'white' if (i < n and display[i, j] > thresh) else 'black'
            ax.text(j, i, f'{int(val):,}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)
    ax.set_xlabel(f'Predicted  [{col_label}]', fontsize=11)
    ax.set_ylabel(f'True  [{row_label}]', fontsize=11)
    ax.set_title(f'{row_label} vs {col_label}   OA = {oa:.4f}', fontsize=12)

    # Draw a dashed line separating the nodata row
    if has_nodata:
        ax.axhline(n - 0.5, color='black', linewidth=1.2, linestyle='--')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def accumulate_nodata_row(row: np.ndarray, true_arr: np.ndarray,
                          pred_arr: np.ndarray, n_classes: int) -> None:
    """
    Accumulate predicted-class counts for pixels where GT == 255 (nodata).
    Only counts pixels where pred_arr has a valid class (< n_classes).
    """
    mask = (true_arr == 255) & (pred_arr < n_classes)
    p    = pred_arr[mask].astype(np.int64)
    row += np.bincount(p, minlength=n_classes)


def compute_iou(cm: np.ndarray) -> dict:
    """
    Compute per-class IoU and mIoU from a confusion matrix.

    IoU_c = CM[c,c] / (CM[c,:].sum() + CM[:,c].sum() - CM[c,c])
    Returns {'oa': float, 'miou': float, 'iou': [iou_c0, iou_c1, ...]}
    """
    n     = cm.shape[0]
    total = cm.sum()
    oa    = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    ious  = []
    for c in range(n):
        tp    = cm[c, c]
        denom = cm[c, :].sum() + cm[:, c].sum() - tp
        ious.append(float(tp / denom) if denom > 0 else 0.0)
    return {'oa': oa, 'miou': float(np.mean(ious)), 'iou': ious}


# ── Per-family worker ─────────────────────────────────────────────────────────

def run_experiment_family(args: tuple) -> list:
    """
    Process all 4 stages of one experiment family sequentially.
    Returns list of metric dicts, one per stage.
    """
    exp_config, gpu_id = args
    exp_base    = exp_config['base']
    inp_mode    = exp_config['inp_mode']
    num_classes = exp_config['num_classes']
    n_cls       = num_classes
    cls_names   = CLASS_NAMES[n_cls]

    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    inp_func = get_inp_func(inp_mode)
    results  = []

    for stage in STAGES:
        exp_name  = f'{exp_base}_stage{stage}'
        out_stage = os.path.join(OUT_DIR, exp_name)
        os.makedirs(out_stage, exist_ok=True)

        print(f'[{exp_name}] loading model on {device} ...', flush=True)

        net = load_model(exp_base, stage, num_classes, inp_mode, device)

        cm_gt_fmask = np.zeros((n_cls, n_cls), dtype=np.int64)
        cm_gt_model = np.zeros((n_cls, n_cls), dtype=np.int64)
        cm_fm_model = np.zeros((n_cls, n_cls), dtype=np.int64)
        # nodata rows: what Fmask/Model predicted for GT=nodata pixels
        nd_gt_fmask = np.zeros(n_cls, dtype=np.int64)
        nd_gt_model = np.zeros(n_cls, dtype=np.int64)

        for scene_id in GT_SCENES:
            print(f'  [{exp_name}] {scene_id}', flush=True)
            prepared_scene = os.path.join(PREPARED_DIR, scene_id)

            spectral = load_scene_bands(prepared_scene)         # (H, W, 8) float32
            cfmask   = load_cfmask(prepared_scene)              # (H, W) uint8
            gt       = load_gt_labels(LABELS_DIR, scene_id)    # (H, W) uint8
            pred     = run_scene_inference(net, inp_func, spectral, device)  # (H, W) uint8

            accumulate_cm(cm_gt_fmask, gt, cfmask, n_cls)
            accumulate_cm(cm_gt_model, gt, pred,   n_cls)
            accumulate_cm(cm_fm_model, cfmask, pred, n_cls)
            accumulate_nodata_row(nd_gt_fmask, gt, cfmask, n_cls)
            accumulate_nodata_row(nd_gt_model, gt, pred,   n_cls)

        # Free GPU memory before loading next stage
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save PNG confusion matrices
        # GT vs * : include nodata row (GT-nodata pixels excluded from OA/IoU)
        save_confusion_matrix_png(
            cm_gt_fmask, 'GT', 'Fmask', cls_names,
            os.path.join(out_stage, 'gt_vs_fmask.png'),
            nodata_row=nd_gt_fmask)
        save_confusion_matrix_png(
            cm_gt_model, 'GT', exp_base, cls_names,
            os.path.join(out_stage, f'gt_vs_{exp_base}.png'),
            nodata_row=nd_gt_model)
        # Fmask vs Model : no nodata row (both always predict a class)
        save_confusion_matrix_png(
            cm_fm_model, 'Fmask', exp_base, cls_names,
            os.path.join(out_stage, f'fmask_vs_{exp_base}.png'))

        # Compute metrics
        m_gf = compute_iou(cm_gt_fmask)
        m_gm = compute_iou(cm_gt_model)
        m_fm = compute_iou(cm_fm_model)

        row = {
            'exp_name':    exp_name,
            'num_classes': n_cls,
            # GT vs Fmask
            'gf_oa':          m_gf['oa'],
            'gf_miou':        m_gf['miou'],
            'gf_nocloud_iou': m_gf['iou'][0],
            'gf_cloud_iou':   m_gf['iou'][1],
            'gf_shadow_iou':  m_gf['iou'][2] if n_cls == 3 else None,
            # GT vs Model
            'gm_oa':          m_gm['oa'],
            'gm_miou':        m_gm['miou'],
            'gm_nocloud_iou': m_gm['iou'][0],
            'gm_cloud_iou':   m_gm['iou'][1],
            'gm_shadow_iou':  m_gm['iou'][2] if n_cls == 3 else None,
            # Fmask vs Model
            'fm_oa':          m_fm['oa'],
            'fm_miou':        m_fm['miou'],
            'fm_nocloud_iou': m_fm['iou'][0],
            'fm_cloud_iou':   m_fm['iou'][1],
            'fm_shadow_iou':  m_fm['iou'][2] if n_cls == 3 else None,
        }
        results.append(row)

        print(
            f'  [{exp_name}]  GT/Fmask → OA={m_gf["oa"]:.4f} mIoU={m_gf["miou"]:.4f} '
            f'nocloud={m_gf["iou"][0]:.4f} cloud={m_gf["iou"][1]:.4f}'
            + (f' shadow={m_gf["iou"][2]:.4f}' if n_cls == 3 else ''),
            flush=True)
        print(
            f'  [{exp_name}]  GT/Model  → OA={m_gm["oa"]:.4f} mIoU={m_gm["miou"]:.4f} '
            f'nocloud={m_gm["iou"][0]:.4f} cloud={m_gm["iou"][1]:.4f}'
            + (f' shadow={m_gm["iou"][2]:.4f}' if n_cls == 3 else ''),
            flush=True)
        print(
            f'  [{exp_name}]  Fm/Model  → OA={m_fm["oa"]:.4f} mIoU={m_fm["miou"]:.4f} '
            f'nocloud={m_fm["iou"][0]:.4f} cloud={m_fm["iou"][1]:.4f}'
            + (f' shadow={m_fm["iou"][2]:.4f}' if n_cls == 3 else ''),
            flush=True)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n_gpus    = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args_list = [
        (exp_config, (i % n_gpus) if n_gpus > 0 else None)
        for i, exp_config in enumerate(EXPERIMENTS)
    ]

    print(f'=== Validation confusion matrix evaluation ===')
    print(f'Experiments: {len(EXPERIMENTS)}  (4 stages each = {len(EXPERIMENTS)*4} models)')
    print(f'Scenes: {len(GT_SCENES)}')
    print(f'GPUs: {n_gpus}  |  Workers: {len(EXPERIMENTS)}')
    print(f'Output: {OUT_DIR}')
    print('==============================================', flush=True)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(EXPERIMENTS)) as pool:
        all_results = pool.map(run_experiment_family, args_list)

    # Flatten results
    flat = [row for family in all_results for row in family]

    # Write comprehensive CSV
    csv_path = os.path.join(OUT_DIR, 'summary_metrics.csv')
    fieldnames = [
        'exp_name', 'num_classes',
        # GT vs Fmask
        'gf_oa', 'gf_miou', 'gf_nocloud_iou', 'gf_cloud_iou', 'gf_shadow_iou',
        # GT vs Model
        'gm_oa', 'gm_miou', 'gm_nocloud_iou', 'gm_cloud_iou', 'gm_shadow_iou',
        # Fmask vs Model
        'fm_oa', 'fm_miou', 'fm_nocloud_iou', 'fm_cloud_iou', 'fm_shadow_iou',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat)

    # Also keep legacy OA-only CSV for backwards compat
    oa_path = os.path.join(OUT_DIR, 'summary_oa.csv')
    with open(oa_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exp_name', 'oa_gt_fmask', 'oa_gt_model', 'oa_fmask_model'])
        for r in flat:
            w.writerow([r['exp_name'], r['gf_oa'], r['gm_oa'], r['fm_oa']])

    print(f'\n=== Done! ===')
    print(f'CSV  : {csv_path}')
    print(f'PNGs : {OUT_DIR}/<exp>_stage<N>/')

    # Print summary table
    hdr = f'{"exp_name":<40s}  {"gf_mIoU":>8s}  {"gf_cloud":>8s}  {"gm_mIoU":>8s}  {"gm_cloud":>8s}  {"fm_mIoU":>8s}  {"fm_cloud":>8s}'
    print(f'\n{hdr}')
    print('-' * len(hdr))
    for r in flat:
        print(f'{r["exp_name"]:<40s}  '
              f'{r["gf_miou"]:8.4f}  {r["gf_cloud_iou"]:8.4f}  '
              f'{r["gm_miou"]:8.4f}  {r["gm_cloud_iou"]:8.4f}  '
              f'{r["fm_miou"]:8.4f}  {r["fm_cloud_iou"]:8.4f}')


if __name__ == '__main__':
    main()
