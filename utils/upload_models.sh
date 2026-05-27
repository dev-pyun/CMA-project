#!/bin/bash
# upload_models.sh — exp_data/ 아래 실험별로 model_best.pth + log/ 를 HF에 개별 업로드
# hf upload CLI를 파일 하나씩 호출해서 CLOSE-WAIT hang 문제를 우회
#
# Usage:
#   ./utils/upload_models.sh dev-pyun/CMA-models

set -e

REPO=${1:?"Usage: $0 <hf_repo_id>"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(dirname "$SCRIPT_DIR")/exp_data"

echo "============================="
echo "Uploading models → ${REPO}"
echo "Exp root: ${EXP_ROOT}"
echo "============================="

for exp_dir in $(ls -d "${EXP_ROOT}"/*/  | sort); do
    exp_name=$(basename "$exp_dir")

    # model_best.pth
    pth="${exp_dir}model/model_best.pth"
    if [ -f "$pth" ]; then
        hf_path="${exp_name}/model/model_best.pth"
        size_mb=$(du -m "$pth" | cut -f1)
        echo ""
        echo "[${exp_name}] Uploading model_best.pth (${size_mb}MB)..."
        hf upload "${REPO}" "$pth" "$hf_path" --repo-type model \
            && echo "  ✓ ${hf_path}" \
            || echo "  ✗ FAILED: ${hf_path}"
    fi

    # log/ 파일들
    log_dir="${exp_dir}log"
    if [ -d "$log_dir" ]; then
        for log_file in "$log_dir"/*; do
            [ -f "$log_file" ] || continue
            log_name=$(basename "$log_file")
            hf_path="${exp_name}/log/${log_name}"
            echo "  Uploading log/${log_name}..."
            hf upload "${REPO}" "$log_file" "$hf_path" --repo-type model \
                && echo "  ✓ ${hf_path}" \
                || echo "  ✗ FAILED: ${hf_path}"
        done
    fi
done

echo ""
echo "============================="
echo "Done → ${REPO}"
echo "============================="
