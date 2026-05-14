"""
Download the SyRIP (Synthetic and Real Infant Pose) dataset.

Steps:
  1. pip install gdown
  2. Get the Google Drive links from the official repo:
     https://github.com/ostadabbas/Infant-Pose-Estimation
     Look for "SyRIP Dataset Download" section in the README.
  3. Paste the file IDs below and run this script.

Expected output layout:
  <project_root>/syrip_raw/
  ├── images/
  │   ├── train/
  │   └── val/
  └── annotations/
      ├── person_keypoints_train.json
      └── person_keypoints_val.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zipfile
from config import SYRIP_RAW_DIR

try:
    import gdown
except ImportError:
    raise SystemExit("Run:  pip install gdown  then re-run this script.")

# ── Fill these in from the repo README ────────────────────────────────────────
TRAIN_IMAGES_ID = "PASTE_GDRIVE_ID_HERE"   # zip of train images
VAL_IMAGES_ID   = "PASTE_GDRIVE_ID_HERE"   # zip of val images
TRAIN_ANNOT_ID  = "PASTE_GDRIVE_ID_HERE"   # person_keypoints_train.json
VAL_ANNOT_ID    = "PASTE_GDRIVE_ID_HERE"   # person_keypoints_val.json
# ─────────────────────────────────────────────────────────────────────────────

OUT       = SYRIP_RAW_DIR
ANNOT_DIR = OUT / "annotations"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)


def gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def download_and_extract(file_id: str, dest_dir: Path, filename: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / filename
    print(f"\nDownloading {filename} ...")
    gdown.download(gdrive_url(file_id), str(out_path), quiet=False)
    if filename.endswith(".zip"):
        print(f"Extracting {filename} ...")
        with zipfile.ZipFile(out_path, "r") as z:
            z.extractall(dest_dir)
        out_path.unlink()


if __name__ == "__main__":
    if "PASTE_GDRIVE_ID_HERE" in [TRAIN_IMAGES_ID, VAL_IMAGES_ID, TRAIN_ANNOT_ID, VAL_ANNOT_ID]:
        print(
            "\n[!] Fill in the Google Drive file IDs first.\n"
            "    Find them at: https://github.com/ostadabbas/Infant-Pose-Estimation\n"
        )
    else:
        download_and_extract(TRAIN_IMAGES_ID, OUT / "images" / "train", "train_images.zip")
        download_and_extract(VAL_IMAGES_ID,   OUT / "images" / "val",   "val_images.zip")
        gdown.download(gdrive_url(TRAIN_ANNOT_ID), str(ANNOT_DIR / "person_keypoints_train.json"), quiet=False)
        gdown.download(gdrive_url(VAL_ANNOT_ID),   str(ANNOT_DIR / "person_keypoints_val.json"),   quiet=False)
        print("\nDone. Run data/convert_syrip_to_yolo.py next.")
