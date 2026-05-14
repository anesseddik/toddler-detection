"""
Fine-tune YOLOv8n-pose on the SyRIP toddler keypoint dataset.

Prerequisites:
  1. python data/download_syrip.py      (fill in Google Drive IDs first)
  2. python data/convert_syrip_to_yolo.py

Keypoints detected (17 COCO joints):
  nose, eyes, ears, shoulders, elbows, WRISTS, hips, knees, ankles
  -> wrist indices 9 (left) and 10 (right) are used later for danger detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import multiprocessing
from ultralytics import YOLO
from config import DATASET_POSE_DIR, RUNS_DIR

DATA   = str(DATASET_POSE_DIR / "data.yaml")
MODEL  = "yolov8n-pose.pt"   # nano-pose; swap to yolov8s-pose.pt for more accuracy
EPOCHS = 100
IMGSZ  = 640
BATCH  = 8
NAME   = "toddler_pose"


def main() -> None:
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(RUNS_DIR),
        name=NAME,
        patience=15,
        save=True,
        plots=True,
        workers=0,      # required on Windows
        amp=False,      # disable AMP on CPU-only machines
        exist_ok=True,
        flipud=0.1,
        degrees=15,
        translate=0.1,
        scale=0.3,
    )

    best = str(RUNS_DIR / NAME / "weights" / "best.pt")
    model = YOLO(best)
    metrics = model.val(data=DATA)
    print(f"\nPose mAP50:    {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    print(f"\nBest weights saved to: {best}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
