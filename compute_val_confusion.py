"""
Validation confusion matrix evaluation for all 20 trained models.

Runs 5 experiment families in parallel (one process each); each process
evaluates 4 stages (stage0→3) sequentially on the 8 GT-labelled scenes.

Output per model (one model = one stage of one experiment):
  val_confusion/{exp_base}_stage{N}/gt_vs_fmask.png
  val_confusion/{exp_base}_stage{N}/gt_vs_{exp_base}.png
  val_confusion/{exp_base}_stage{N}/fmask_vs_{exp_base}.png
  val_confusion/summary_oa.csv
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
                               class_names: list, save_path: str) -> None:
    """Save confusion matrix as PNG with per-cell counts and row-normalised %."""
    n     = cm.shape[0]
    total = cm.sum()
    oa    = cm.diagonal().sum() / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(4 + n * 1.5, 3 + n * 1.5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(n):
        row_sum = cm[i].sum()
        for j in range(n):
            pct = cm[i, j] / row_sum * 100 if row_sum > 0 else 0.0
            ax.text(j, i, f'{cm[i, j]:,}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel(f'Predicted  [{col_label}]', fontsize=11)
    ax.set_ylabel(f'True  [{row_label}]', fontsize=11)
    ax.set_title(f'{row_label} vs {col_label}   OA = {oa:.4f}', fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Per-family worker ─────────────────────────────────────────────────────────

def run_experiment_family(args: tuple) -> list:
    """
    Process all 4 stages of one experiment family sequentially.
    Returns list of (exp_name, oa_gt_fmask, oa_gt_model, oa_fmask_model) tuples.
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

        # Free GPU memory before loading next stage
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save PNG confusion matrices
        save_confusion_matrix_png(
            cm_gt_fmask, 'GT', 'Fmask', cls_names,
            os.path.join(out_stage, 'gt_vs_fmask.png'))
        save_confusion_matrix_png(
            cm_gt_model, 'GT', exp_base, cls_names,
            os.path.join(out_stage, f'gt_vs_{exp_base}.png'))
        save_confusion_matrix_png(
            cm_fm_model, 'Fmask', exp_base, cls_names,
            os.path.join(out_stage, f'fmask_vs_{exp_base}.png'))

        def _oa(cm):
            s = cm.sum()
            return float(cm.diagonal().sum() / s) if s > 0 else 0.0

        oa_gf = _oa(cm_gt_fmask)
        oa_gm = _oa(cm_gt_model)
        oa_fm = _oa(cm_fm_model)
        results.append((exp_name, oa_gf, oa_gm, oa_fm))

        print(f'  [{exp_name}] OA  gt/fmask={oa_gf:.4f}  '
              f'gt/model={oa_gm:.4f}  fmask/model={oa_fm:.4f}', flush=True)

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

    # Flatten and write CSV
    flat     = [row for family in all_results for row in family]
    csv_path = os.path.join(OUT_DIR, 'summary_oa.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exp_name', 'oa_gt_fmask', 'oa_gt_model', 'oa_fmask_model'])
        w.writerows(flat)

    print(f'\n=== Done! ===')
    print(f'CSV  : {csv_path}')
    print(f'PNGs : {OUT_DIR}/<exp>_stage<N>/')

    # Print summary table
    print('\n{:<40s}  {:>10s}  {:>10s}  {:>10s}'.format(
        'exp_name', 'gt/fmask', 'gt/model', 'fmask/model'))
    print('-' * 75)
    for exp_name, oa_gf, oa_gm, oa_fm in flat:
        print(f'{exp_name:<40s}  {oa_gf:10.4f}  {oa_gm:10.4f}  {oa_fm:10.4f}')


if __name__ == '__main__':
    main()
