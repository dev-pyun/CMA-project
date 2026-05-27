"""
utils/upload_to_hf.py — Hugging Face Hub 업로드 유틸리티

모델 (model_best.pth + log/) 또는 패치 데이터 (TRAIN_ZARR, VALIDATION_ZARR)를
Hugging Face Hub에 업로드합니다.

사용 전 준비:
    1. https://huggingface.co/settings/tokens 에서 토큰 발급 (write 권한)
    2. huggingface-cli login  또는  --token 플래그로 전달

Usage:
    # 모델 업로드 (model_best.pth + log/)
    python utils/upload_to_hf.py --mode models \\
        --repo dev-pyun/CMA-models \\
        --token hf_xxxx

    # 패치 데이터 업로드 (TRAIN_ZARR + VALIDATION_ZARR)
    python utils/upload_to_hf.py --mode data \\
        --repo dev-pyun/CMA-patches \\
        --token hf_xxxx
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dir_paths import TRAIN_ZARR_PATH, VALID_ZARR_PATH

EXP_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exp_data')


def upload_models(api, repo_id: str) -> None:
    """
    exp_data/ 아래 각 실험의 model_best.pth + log/ 폴더를 upload_folder로 업로드.
    파일 하나씩 commit하면 연결이 끊기는 문제가 있어 폴더 단위로 묶어서 올림.
    HF 경로 구조: {exp_name}/model/model_best.pth
                  {exp_name}/log/{파일들}
    """
    import tempfile, shutil

    exp_root = Path(EXP_DATA_PATH)
    uploaded = 0

    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        best_pth = exp_dir / 'model' / 'model_best.pth'
        log_dir = exp_dir / 'log'
        has_model = best_pth.exists()
        has_log = log_dir.exists()

        if not has_model and not has_log:
            continue

        print(f'\n  Uploading {exp_dir.name}/ ...')

        # 임시 디렉토리에 model_best.pth + log/ 복사 후 upload_folder
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if has_model:
                (tmp_path / 'model').mkdir()
                shutil.copy2(best_pth, tmp_path / 'model' / 'model_best.pth')
            if has_log:
                shutil.copytree(log_dir, tmp_path / 'log')

            api.upload_folder(
                folder_path=str(tmp_path),
                path_in_repo=exp_dir.name,
                repo_id=repo_id,
                repo_type='model',
            )
            uploaded += 1

    print(f'\nDone. Uploaded {uploaded} experiments → {repo_id}')


def upload_data(api, repo_id: str) -> None:
    """
    TRAIN_ZARR/ 와 VALIDATION_ZARR/ 전체를 데이터셋 repo에 업로드.
    stage_*.txt 파일은 제외 (재생성 가능).
    upload_large_folder는 path_in_repo 미지원 → 부모 폴더(data/)에서
    allow_patterns으로 각 split을 필터링해서 올림.
    """
    data_root = Path(TRAIN_ZARR_PATH).parent  # data/

    for split, src_path in [('TRAIN_ZARR', TRAIN_ZARR_PATH),
                             ('VALIDATION_ZARR', VALID_ZARR_PATH)]:
        src = Path(src_path)
        if not src.exists():
            print(f'  [SKIP] {src_path} not found.')
            continue

        print(f'\nUploading {split}/ ({src_path})...')
        api.upload_large_folder(
            folder_path=str(data_root),
            repo_id=repo_id,
            repo_type='dataset',
            allow_patterns=[f'{split}/**'],
            ignore_patterns=['**/stage_*.txt'],
        )
        print(f'  Done → {repo_id}/{split}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Upload models or patches to HF Hub')
    parser.add_argument('--mode', choices=['models', 'data'], required=True,
                        help='models: exp_data best.pth + logs / data: TRAIN+VALID ZARR')
    parser.add_argument('--repo', required=True,
                        help='HF repo id, e.g. dev-pyun/CMA-models')
    parser.add_argument('--token', default=None,
                        help='HF write token (or use huggingface-cli login)')
    args = parser.parse_args()

    from huggingface_hub import HfApi, login
    if args.token:
        login(token=args.token)

    api = HfApi()

    # repo 없으면 자동 생성
    repo_type = 'model' if args.mode == 'models' else 'dataset'
    try:
        api.repo_info(repo_id=args.repo, repo_type=repo_type)
        print(f'Repo {args.repo} already exists.')
    except Exception:
        print(f'Creating repo {args.repo} ({repo_type})...')
        api.create_repo(repo_id=args.repo, repo_type=repo_type, private=True)

    if args.mode == 'models':
        print(f'\nUploading models → {args.repo}')
        upload_models(api, args.repo)
    else:
        print(f'\nUploading patch data → {args.repo}')
        upload_data(api, args.repo)


if __name__ == '__main__':
    main()
