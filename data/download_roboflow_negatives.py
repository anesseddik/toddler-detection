"""
Download hard-negative image sets from Roboflow Universe and add them
to the detection training set with empty label files (background class).

Categories covered: pets (cats/dogs), furniture, indoor rooms, toys.

Setup:
    pip install roboflow
    python data/download_roboflow_negatives.py --api-key YOUR_KEY

API key: https://app.roboflow.com/ -> top-right avatar -> Settings -> API Keys

To change/add datasets: edit the DATASETS list below.
Browse slugs at: https://universe.roboflow.com
"""

import sys
import argparse
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from roboflow import Roboflow
from config import DATASET_DETECT_DIR

TRAIN_IMGS = DATASET_DETECT_DIR / "train" / "images"
TRAIN_LBLS = DATASET_DETECT_DIR / "train" / "labels"
TRAIN_IMGS.mkdir(parents=True, exist_ok=True)
TRAIN_LBLS.mkdir(parents=True, exist_ok=True)

# Any image whose label file references one of these class names is skipped.
# We never want a human/child image with an empty label — that confuses the model.
PERSON_LABELS = {
    "person", "child", "baby", "toddler", "kid", "infant",
    "human", "people", "man", "woman", "boy", "girl",
}

# ── Datasets to download ──────────────────────────────────────────────────────
# Format: (workspace_slug, project_slug, version, description)
# Verify/change slugs at https://universe.roboflow.com
DATASETS = [
    # Pets — no humans, only cats & dogs
    ("brad-dwyer",             "oxford-pets",               1, "Pets: Oxford IIIT cats & dogs"),
    # Furniture — chairs, tables, sofas, beds, etc.
    ("mokhamed-nagy-u69zl",    "furniture-detection-qiufc", 1, "Furniture detection (8 k imgs)"),
    # Indoor room scenes — walls, floors, windows, doors
    ("research-twzom",         "room-interior",             1, "Indoor room scenes"),
    # Toys — no kids in scene
    ("toys-detection-project", "toys-detection-new-data",   1, "Toys (7 k imgs)"),
]
# ─────────────────────────────────────────────────────────────────────────────


def person_class_ids(data_yaml: Path) -> set:
    """Return class IDs that are person-like, by reading a YOLO data.yaml."""
    if not data_yaml.exists():
        return set()
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(idx) for idx, name in names.items()
            if str(name).lower() in PERSON_LABELS}


def has_person(label_file: Path, bad_ids: set) -> bool:
    """Return True if a YOLO label file contains any person-like class."""
    if not label_file.exists() or not bad_ids:
        return False
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in bad_ids:
            return True
    return False


def add_to_train(img_path: Path, prefix: str) -> bool:
    """Copy one image to the train set with an empty label. Returns True if added."""
    dst_img = TRAIN_IMGS / f"{prefix}_{img_path.name}"
    if dst_img.exists():
        return False
    shutil.copy2(img_path, dst_img)
    (TRAIN_LBLS / dst_img.with_suffix(".txt").name).write_text("")
    return True


def process_dataset(dl_dir: Path, prefix: str) -> tuple:
    """Walk all splits of a downloaded Roboflow dataset and add valid images."""
    data_yaml = dl_dir / "data.yaml"
    bad_ids = person_class_ids(data_yaml)
    if bad_ids:
        print(f"   Filtering out class IDs {bad_ids} (person-like)")

    added = skipped_person = skipped_dup = 0
    for split in ("train", "valid", "test"):
        img_dir = dl_dir / split / "images"
        lbl_dir = dl_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if has_person(lbl, bad_ids):
                skipped_person += 1
                continue
            if add_to_train(img, prefix):
                added += 1
            else:
                skipped_dup += 1

    return added, skipped_person, skipped_dup


def main():
    parser = argparse.ArgumentParser(
        description="Download Roboflow hard negatives into the training set."
    )
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    args = parser.parse_args()

    rf = Roboflow(api_key=args.api_key)
    total_added = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for workspace, project_slug, version, desc in DATASETS:
            print(f"\n── {desc}")
            print(f"   {workspace}/{project_slug}  v{version}")
            try:
                project = rf.workspace(workspace).project(project_slug)
                dataset = project.version(version).download(
                    "yolov8", location=str(tmp_path / project_slug)
                )
                dl_dir = Path(dataset.location)
                prefix = project_slug.replace("-", "_")[:18]
                added, skipped_p, skipped_d = process_dataset(dl_dir, prefix)
                print(f"   Added: {added}  |  Skipped-person: {skipped_p}  |  Skipped-dup: {skipped_d}")
                total_added += added
            except Exception as e:
                print(f"   ERROR: {e}")
                print("   -> Check the workspace/project slug and version number.")

    train_total = len(list(TRAIN_IMGS.glob("*.*")))
    print(f"\n{'─'*55}")
    print(f"  New hard negatives added : {total_added}")
    print(f"  Train images total now   : {train_total}")
    print(f"  Train folder             : {TRAIN_IMGS}")
    print(f"{'─'*55}")
    print("\nNext step: upload dataset to Colab and retrain.")


if __name__ == "__main__":
    main()
