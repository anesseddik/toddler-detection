import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
import multiprocessing
from config import DATASET_DETECT_DIR, RUNS_DIR

DATA   = str(DATASET_DETECT_DIR / "data.yaml")
MODEL  = "yolov8n.pt"   # nano; swap to yolov8s.pt / yolov8m.pt for more capacity
EPOCHS = 50
IMGSZ  = 640
BATCH  = 8              # lower batch for CPU training
NAME   = "toddler_detect"


def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(RUNS_DIR),
        name=NAME,
        patience=10,
        save=True,
        plots=True,
        workers=0,    # required on Windows to avoid multiprocessing crash
        amp=False,    # disable AMP — unstable on CPU-only machines
        exist_ok=True,
    )

    metrics = model.val(
        data=DATA,
        split="test",
        project=str(RUNS_DIR),
        name=NAME + "_test_eval",
    )
    print(f"\nmAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
