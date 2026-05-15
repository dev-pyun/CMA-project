"""
3-class 파이프라인 동작 확인 스크립트.

실제 학습/검증 데이터 일부로 다음 7가지 항목을 검증한다:
  [1] qa_pixel_to_binary()   → 출력 값이 {0, 1, 2, 255} 인지
  [2] Train DataLoader       → 라벨 값 범위 {0, 1, 2, 255}
  [3] Val DataLoader         → 라벨 값 범위 {0, 1, 2, 255}
  [4] Model forward pass     → 출력 shape (B, 3, H, W)
  [5] Loss + backward        → 유한한 loss, gradient 흐름
  [6] MFB weights            → shape (3,), 모두 양수
  [7] encode_label           → pseudo-label 값이 {0, 1, 2, 255}
  [8] Confusion matrix       → shape (3, 3)
  [9] train/val 1 epoch      → 실제 루프 에러 없이 완료

사용:
    conda run -n remote python test_pipeline.py -gpu 0
    conda run -n remote python test_pipeline.py -gpu 0 -ip swirndsindwi
"""

import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── ANSI 색상 ─────────────────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
RESET  = '\033[0m'

PASS = f'{GREEN}[PASS]{RESET}'
FAIL = f'{RED}[FAIL]{RESET}'
INFO = f'{YELLOW}[INFO]{RESET}'

results = []

def check(name, cond, detail=''):
    tag = PASS if cond else FAIL
    msg = f'{tag} {name}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    results.append((name, cond))
    return cond


def get_args():
    p = argparse.ArgumentParser(description='3-class pipeline 동작 확인')
    p.add_argument('-gpu', '--gpu_id', type=int, nargs='+', default=[0])
    p.add_argument('-ip',  '--inp_mode', default='swirndsi')
    p.add_argument('-bs',  '--batch_size', type=int, default=4)
    p.add_argument('-n',   '--n_batches', type=int, default=3,
                   help='테스트에 사용할 배치 수 (기본: 3)')
    return p.parse_args()


def main():
    args = get_args()
    print(f'\n{"="*60}')
    print(f'  3-class pipeline test  |  gpu={args.gpu_id}  inp={args.inp_mode}')
    print(f'{"="*60}\n')

    # ── [1] qa_pixel_to_binary 출력 값 검증 ──────────────────────────
    print('[1] qa_pixel_to_binary() 클래스 값 검증')
    from utils.qa_pixel_mapping import qa_pixel_to_binary, NUM_BINARY_CLASSES

    # 각 비트 개별 테스트
    test_cases = {
        'Fill(bit0)':         (np.array([[1 << 0]], dtype=np.uint16),  {255}),
        'Dilated(bit1)':      (np.array([[1 << 1]], dtype=np.uint16),  {1}),
        'Cirrus(bit2)':       (np.array([[1 << 2]], dtype=np.uint16),  {1}),
        'Cloud(bit3)':        (np.array([[1 << 3]], dtype=np.uint16),  {1}),
        'Shadow(bit4)':       (np.array([[1 << 4]], dtype=np.uint16),  {2}),
        'Snow(bit5)':         (np.array([[1 << 5]], dtype=np.uint16),  {0}),
        'Clear(bit6)':        (np.array([[1 << 6]], dtype=np.uint16),  {0}),
        'Water(bit7)':        (np.array([[1 << 7]], dtype=np.uint16),  {0}),
        'Cloud+Shadow':       (np.array([[(1<<3)|(1<<4)]], dtype=np.uint16), {1}),  # cloud 우선
    }

    qa_ok = True
    for case_name, (qa, expected) in test_cases.items():
        got = set(np.unique(qa_pixel_to_binary(qa)))
        ok = got == expected
        if not ok:
            print(f'  {RED}FAIL{RESET} {case_name}: expected {expected}, got {got}')
            qa_ok = False
        else:
            print(f'  {GREEN}OK{RESET}   {case_name}: {got}')

    check('qa_pixel_to_binary 클래스 값', qa_ok)
    check('NUM_BINARY_CLASSES == 3', NUM_BINARY_CLASSES == 3,
          f'실제값: {NUM_BINARY_CLASSES}')
    print()

    # ── DataLoader 준비 ───────────────────────────────────────────────
    from dataset.patch_dataset import setup_data
    from utils.dir_paths import TRAIN_PATH, VALID_PATH

    print('[2] Train DataLoader 라벨 값 검증')
    try:
        train_loader = setup_data(
            args.batch_size, mode='train', stage=0,
            path=TRAIN_PATH, full=False, aug=False)
        train_iter = iter(train_loader)
        _, labels, _ = next(train_iter)
        lbl_np = labels[:, 0, :, :].numpy()
        unique_vals = set(np.unique(lbl_np).tolist())
        valid_vals  = unique_vals - {255}
        ok = valid_vals.issubset({0, 1, 2})
        check('Train 라벨 값 ⊆ {0,1,2,255}', ok,
              f'발견된 값: {sorted(unique_vals)}')
    except Exception as e:
        check('Train DataLoader 로드', False, str(e))
        train_loader = None
    print()

    print('[3] Val DataLoader 라벨 값 검증')
    try:
        val_loader = setup_data(
            args.batch_size, mode='test', path=VALID_PATH)
        val_iter = iter(val_loader)
        _, val_labels, _ = next(val_iter)
        val_lbl_np = val_labels[:, 0, :, :].numpy()
        val_unique  = set(np.unique(val_lbl_np).tolist())
        val_valid   = val_unique - {255}
        ok = val_valid.issubset({0, 1, 2})
        check('Val 라벨 값 ⊆ {0,1,2,255}', ok,
              f'발견된 값: {sorted(val_unique)}')
    except Exception as e:
        check('Val DataLoader 로드', False, str(e))
        val_loader = None
    print()

    # ── Model 구성 ────────────────────────────────────────────────────
    print('[4] Model forward pass (출력 shape 확인)')
    from network.model import Model, NUM_CLASSES
    from utils.experiment import Experiment

    check('NUM_CLASSES == 3', NUM_CLASSES == 3, f'실제값: {NUM_CLASSES}')

    class _Args:
        exp_name = '__test_3class__'
        stage = 0; full = False; dropout = True; learning_rate = 1e-6
        inp_mode = args.inp_mode; bands = None; indices = None

    exp = Experiment(_Args(), mode='train')
    model = Model(exp, gpu_id=args.gpu_id)

    # dummy forward
    dummy_inp = torch.zeros(2, 17, 258, 258)   # (B, 17, H+2, W+2)
    with torch.no_grad():
        out = model.forward_step(dummy_inp)
    expected_shape = (2, NUM_CLASSES, 258, 258)
    check('출력 shape', tuple(out.shape) == expected_shape,
          f'{tuple(out.shape)}')
    print()

    # ── [5] Loss + backward ───────────────────────────────────────────
    print('[5] Loss 계산 + backward')
    model.network.train()
    dummy_lbl = torch.zeros(2, 258, 258, dtype=torch.long)
    dummy_lbl[0, :, :] = 1   # cloud
    dummy_lbl[1, :, :] = 2   # shadow
    exp.weights = np.ones(NUM_CLASSES, dtype=np.float32)

    out = model.forward_step(dummy_inp)
    import torch.nn.functional as F
    w = torch.from_numpy(exp.weights).float().to(model.device)
    loss = F.cross_entropy(out, dummy_lbl.to(model.device), w, ignore_index=255)

    check('Loss 유한값', torch.isfinite(loss).item(), f'loss={loss.item():.4f}')
    loss_expected_range = 0.5 < loss.item() < 3.0   # ~log(3) ≈ 1.1 for random
    check('Loss 범위 합리적', loss_expected_range, f'{loss.item():.4f}')

    loss.backward()
    grads = [p.grad for p in model.network.parameters() if p.grad is not None]
    check('Gradient 흐름', len(grads) > 0, f'{len(grads)} param에 grad 존재')
    print()

    # ── [6] MFB weights ───────────────────────────────────────────────
    print('[6] MFB weights')
    if train_loader is not None:
        from utils.MFB import get_MFB_weights
        # 처음 몇 배치만 샘플링해서 빠르게 계산
        from torch.utils.data import DataLoader, Subset
        mini_ds = Subset(train_loader.dataset, list(range(min(20, len(train_loader.dataset)))))
        mini_loader = DataLoader(mini_ds, batch_size=args.batch_size)
        weights = get_MFB_weights(mini_loader, num_classes=NUM_CLASSES)
        check('MFB weights shape (3,)', weights.shape == (NUM_CLASSES,),
              f'shape={weights.shape}')
        check('MFB weights 모두 양수', (weights > 0).all(),
              f'weights={weights}')
    else:
        print(f'  {YELLOW}SKIP{RESET} train_loader 없음')
    print()

    # ── [7] encode_label ──────────────────────────────────────────────
    print('[7] encode_label (pseudo-label 값 범위)')
    model.network.eval()
    with torch.no_grad():
        dummy_out = model.forward_step(dummy_inp)
        pseudo = model.encode_label(dummy_out, label_gen=True)
    pseudo_vals = set(pseudo.unique().cpu().numpy().tolist())
    ok = pseudo_vals.issubset({0, 1, 2, 255})
    check('pseudo-label 값 ⊆ {0,1,2,255}', ok,
          f'발견된 값: {pseudo_vals}')
    print()

    # ── [8] Confusion matrix shape ────────────────────────────────────
    print('[8] Confusion matrix shape (3×3)')
    from utils.metrics import Metrics
    device = model.device
    metrics = Metrics(device, num_classes=NUM_CLASSES)
    check('Confusion matrix shape (3,3)',
          tuple(metrics.val_confusion_matrix.shape) == (NUM_CLASSES, NUM_CLASSES),
          f'{tuple(metrics.val_confusion_matrix.shape)}')
    print()

    # ── [9] 실제 train/val 1 에포크 ──────────────────────────────────
    print(f'[9] 실제 train/val 루프 ({args.n_batches} 배치씩)')
    if train_loader is not None and val_loader is not None:
        from utils.MFB import get_MFB_weights

        exp2_args = type(_Args)('_Args2', (), {
            'exp_name': '__test_3class_run__',
            'stage': 0, 'full': False, 'dropout': True,
            'learning_rate': 1e-4, 'inp_mode': args.inp_mode,
            'bands': None, 'indices': None,
        })
        exp2 = Experiment(exp2_args(), mode='train')
        model2 = Model(exp2, gpu_id=args.gpu_id)

        mini_ds   = Subset(train_loader.dataset, list(range(min(args.n_batches * args.batch_size, len(train_loader.dataset)))))
        mini_val  = Subset(val_loader.dataset,   list(range(min(args.n_batches * args.batch_size, len(val_loader.dataset)))))
        mini_train_loader = DataLoader(mini_ds,  batch_size=args.batch_size, collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None)
        mini_val_loader   = DataLoader(mini_val, batch_size=args.batch_size, collate_fn=val_loader.collate_fn   if hasattr(val_loader, 'collate_fn') else None)

        exp2.weights = get_MFB_weights(mini_train_loader, num_classes=NUM_CLASSES)

        train_ok = True
        try:
            model2.network.train()
            for batch_data in mini_train_loader:
                loss = model2.train_step(batch_data)
                if not torch.isfinite(loss):
                    train_ok = False
                    break
        except Exception as e:
            print(f'  {RED}ERROR{RESET} train loop: {e}')
            train_ok = False
        check('Train loop 에러 없음', train_ok)

        val_ok = True
        try:
            model2.network.eval()
            with torch.no_grad():
                for batch_data in mini_val_loader:
                    model2.valid_step(batch_data, mode='test')
        except Exception as e:
            print(f'  {RED}ERROR{RESET} val loop: {e}')
            val_ok = False
        check('Val loop 에러 없음', val_ok)

        if train_ok and val_ok:
            try:
                cm = model2.metrics.val_confusion_matrix.cpu().numpy()
                train_m, val_m, class_iou = model2.metrics.aggregate_metrics(0)
                check('mIoU 계산 성공', True,
                      f'mIoU={val_m["mIoU"]:.4f}  '
                      f'IoU: ' + '  '.join(f'cls{c}={v:.3f}' for c, v in class_iou.items()))
            except Exception as e:
                check('mIoU 계산 성공', False, str(e))
    else:
        print(f'  {YELLOW}SKIP{RESET} DataLoader 없음')
    print()

    # ── 최종 결과 ─────────────────────────────────────────────────────
    print('='*60)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f'결과: {GREEN}{n_pass} PASS{RESET}  /  {RED}{n_fail} FAIL{RESET}  (총 {len(results)})')
    if n_fail > 0:
        print(f'\n{RED}실패 항목:{RESET}')
        for name, ok in results:
            if not ok:
                print(f'  - {name}')
    print('='*60)

    # 테스트 exp 폴더 정리
    import shutil
    for d in ['__test_3class__', '__test_3class_run__']:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_data', d)
        if os.path.exists(p):
            shutil.rmtree(p)


if __name__ == '__main__':
    main()
