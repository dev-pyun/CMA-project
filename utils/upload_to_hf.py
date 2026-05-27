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
    exp_data/ 아래 각 실험의 model_best.pth + log/ 폴더를 업로드.
    HF 경로 구조: {exp_name}/model/model_best.pth
                  {exp_name}/log/{파일들}
    """
    exp_root = Path(EXP_DATA_PATH)
    uploaded = 0

    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        # model_best.pth
        best_pth = exp_dir / 'model' / 'model_best.pth'
        if best_pth.exists():
            hf_path = f'{exp_dir.name}/model/model_best.pth'
            print(f'  Uploading {hf_path} ({best_pth.stat().st_size / 1e6:.1f} MB)...')
            api.upload_file(
                path_or_fileobj=str(best_pth),
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type='model',
            )
            uploaded += 1

        # log/ 폴더
        log_dir = exp_dir / 'log'
        if log_dir.exists():
            for log_file in log_dir.iterdir():
                if log_file.is_file():
                    hf_path = f'{exp_dir.name}/log/{log_file.name}'
                    print(f'  Uploading {hf_path}...')
                    api.upload_file(
                        path_or_fileobj=str(log_file),
                        path_in_repo=hf_path,
                        repo_id=repo_id,
                        repo_type='model',
                    )
                    uploaded += 1

    print(f'\nDone. Uploaded {uploaded} files → {repo_id}')


def upload_data(api, repo_id: str) -> None:
    """
    TRAIN_ZARR/ 와 VALIDATION_ZARR/ 전체를 데이터셋 repo에 업로드.
    stage_*.txt 파일은 제외 (재생성 가능).
    """
    for split, src_path in [('TRAIN_ZARR', TRAIN_ZARR_PATH),
                             ('VALIDATION_ZARR', VALID_ZARR_PATH)]:
        src = Path(src_path)
        if not src.exists():
            print(f'  [SKIP] {src_path} not found.')
            continue

        print(f'\nUploading {split}/ ({src_path})...')
        # stage_*.txt 제외하고 업로드
        ignore_patterns = ['stage_*.txt']
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=split,
            repo_id=repo_id,
            repo_type='dataset',
            ignore_patterns=ignore_patterns,
            multi_commits=True,        # 대용량 폴더는 여러 커밋으로 분할
            multi_commits_verbose=True,
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
