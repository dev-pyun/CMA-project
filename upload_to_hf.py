"""Upload TRAIN_ZARR and VALIDATION_ZARR to HuggingFace dev-pyun/CMA-patches"""
from huggingface_hub import HfApi

REPO_ID = "dev-pyun/CMA-patches"
LOCAL_DATA_DIR = "/home/pyuncb/src/data"

api = HfApi()

print(f"=== Starting upload to {REPO_ID} ===")
print(f"Source: {LOCAL_DATA_DIR}")
print("Uploading TRAIN_ZARR/** and VALIDATION_ZARR/**")

api.upload_large_folder(
    repo_id=REPO_ID,
    folder_path=LOCAL_DATA_DIR,
    repo_type="dataset",
    allow_patterns=["TRAIN_ZARR/**", "VALIDATION_ZARR/**"],
    num_workers=4,
)

print("Upload complete.")
