"""
Convert SyRIP COCO-format keypoint annotations -> YOLO pose format.

COCO keypoint order (17 joints):
  0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
  5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
  9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
  13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

YOLO pose label line format (all values normalised to [0,1]):
  class  cx  cy  bw  bh  x0 y0 v0  x1 y1 v1  ...  x16 y16 v16

Run after data/download_syrip.py:
  python data/convert_syrip_to_yolo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import shutil
from collections import defaultdict
from config import SYRIP_RAW_DIR, DATASET_POSE_DIR

SRC_ROOT = SYRIP_RAW_DIR
DST_ROOT = DATASET_POSE_DIR
SPLITS   = {
    "train": "person_keypoints_train.json",
    "val":   "person_keypoints_val.json",
}
N_KPT    = 17
CLASS_ID = 0   # single class: toddler


def coco_to_yolo_pose(ann: dict, img_w: int, img_h: int) -> str:
    x, y, bw, bh = ann["bbox"]
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    bw /= img_w
    bh /= img_h

    kpts     = ann["keypoints"]
    kpt_parts = []
    for i in range(N_KPT):
        kx = kpts[i * 3]     / img_w
        ky = kpts[i * 3 + 1] / img_h
        kv = kpts[i * 3 + 2]   # 0=not labeled, 1=hidden, 2=visible
        kpt_parts.append(f"{kx:.6f} {ky:.6f} {kv}")

    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_parts)


def convert_split(split: str, json_name: str) -> None:
    annot_path = SRC_ROOT / "annotations" / json_name
    img_src    = SRC_ROOT / "images" / split

    img_dst = DST_ROOT / "images" / split
    lbl_dst = DST_ROOT / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {split} split  ({annot_path.name}) ...")
    with open(annot_path, "r") as f:
        data = json.load(f)

    id_to_img = {img["id"]: img for img in data["images"]}

    img_anns: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("num_keypoints", 0) == 0:
            continue
        img_anns[ann["image_id"]].append(ann)

    copied  = 0
    skipped = 0
    for img_id, anns in img_anns.items():
        img_info = id_to_img[img_id]
        img_file = img_src / img_info["file_name"]
        if not img_file.exists():
            skipped += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        lines = [coco_to_yolo_pose(a, img_w, img_h) for a in anns]

        dst_img = img_dst / img_info["file_name"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, dst_img)

        lbl_file = lbl_dst / (Path(img_info["file_name"]).stem + ".txt")
        lbl_file.parent.mkdir(parents=True, exist_ok=True)
        lbl_file.write_text("\n".join(lines))
        copied += 1

    print(f"  {copied} images converted, {skipped} skipped (image not found).")


if __name__ == "__main__":
    for split, json_name in SPLITS.items():
        convert_split(split, json_name)

    print(f"\nDataset ready at: {DST_ROOT}")
    print("Next step: run  python training/train_pose.py")
