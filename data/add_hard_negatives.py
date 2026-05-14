"""
Copies all frames from adult_negatives/ into the training set with empty
label files, teaching the model to suppress false positives in those scenes.

Run once before retraining:
  python data/add_hard_negatives.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil
from config import ADULT_NEG_DIR, DATASET_DETECT_DIR

TRAIN_IMGS = DATASET_DETECT_DIR / "train" / "images"
TRAIN_LBLS = DATASET_DETECT_DIR / "train" / "labels"

TRAIN_IMGS.mkdir(parents=True, exist_ok=True)
TRAIN_LBLS.mkdir(parents=True, exist_ok=True)

frames = sorted(ADULT_NEG_DIR.glob("*.jpg"))
if not frames:
    print("No frames found in adult_negatives/. Run extract_hard_negatives.py first.")
    sys.exit(1)

added    = 0
skipped  = 0
for img_path in frames:
    dst_img = TRAIN_IMGS / img_path.name
    dst_lbl = TRAIN_LBLS / (img_path.stem + ".txt")

    if dst_img.exists():
        skipped += 1
        continue

    shutil.copy2(img_path, dst_img)
    dst_lbl.write_text("")   # empty label = background image
    added += 1

print(f"Added   : {added} hard negative frames to training set")
print(f"Skipped : {skipped} already present")
print(f"Train images folder now has: {len(list(TRAIN_IMGS.glob('*.*')))} total images")
print(f"\nNext step: upload dataset to Colab and retrain.")
