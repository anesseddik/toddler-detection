"""
Splits 20% of the train set into a validation set.
Run once before training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil
import random
from config import DATASET_DETECT_DIR

SEED      = 42
VAL_RATIO = 0.2

TRAIN_IMGS = DATASET_DETECT_DIR / "train" / "images"
TRAIN_LBLS = DATASET_DETECT_DIR / "train" / "labels"
VAL_IMGS   = DATASET_DETECT_DIR / "val"   / "images"
VAL_LBLS   = DATASET_DETECT_DIR / "val"   / "labels"

VAL_IMGS.mkdir(parents=True, exist_ok=True)
VAL_LBLS.mkdir(parents=True, exist_ok=True)

all_images = sorted(TRAIN_IMGS.glob("*.*"))
random.seed(SEED)
random.shuffle(all_images)

val_count  = int(len(all_images) * VAL_RATIO)
val_images = all_images[:val_count]

moved   = 0
skipped = 0
for img_path in val_images:
    lbl_path = TRAIN_LBLS / (img_path.stem + ".txt")
    if not lbl_path.exists():
        skipped += 1
        continue
    shutil.move(str(img_path), VAL_IMGS / img_path.name)
    shutil.move(str(lbl_path), VAL_LBLS / lbl_path.name)
    moved += 1

print(f"Moved {moved} images to val/ (skipped {skipped} without labels)")
print(f"Train remaining: {len(list(TRAIN_IMGS.glob('*.*')))}")
print(f"Val:             {len(list(VAL_IMGS.glob('*.*')))}")
print(f"Test:            {len(list((DATASET_DETECT_DIR / 'test' / 'images').glob('*.*')))}")
