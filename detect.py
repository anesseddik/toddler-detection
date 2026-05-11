from ultralytics import YOLO
import sys

MODEL  = "C:/pfe/toddler_detect_s100_best.pt"
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"   # default: webcam

model = YOLO(MODEL)

results = model.predict(
    source=SOURCE,
    conf=0.4,          # minimum confidence threshold
    save=True,         # saves output to runs/detect/
    show=True,         # shows live window (close with Q)
    line_width=2,
)

print(f"\nResults saved to: {results[0].save_dir}")
