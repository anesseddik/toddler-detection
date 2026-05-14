import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from ultralytics import YOLO
from config import FINAL_DETECT_MODEL

MODEL         = str(FINAL_DETECT_MODEL)
SOURCE        = sys.argv[1] if len(sys.argv) > 1 else "0"   # default: webcam
DISPLAY_WIDTH = 960   # resize window to this width (keep aspect ratio)

model = YOLO(MODEL)

frames_with_detection = 0
total_detections      = 0
confidences           = []
total_frames          = 0
save_dir              = None

for r in model.predict(source=SOURCE, conf=0.4, save=True, stream=True, line_width=2):
    total_frames += 1
    n = len(r.boxes)
    if n > 0:
        frames_with_detection += 1
        total_detections      += n
        confidences.extend(r.boxes.conf.tolist())

    if save_dir is None:
        save_dir = r.save_dir

    frame = r.plot()
    h, w  = frame.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.imshow("Toddler Detector  [Q to quit]",
               cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print(f"\n{'─'*45}")
print(f"  Video:             {SOURCE}")
print(f"  Total frames:      {total_frames}")
print(f"  Frames with det.:  {frames_with_detection}  ({100*frames_with_detection/max(total_frames,1):.1f}%)")
print(f"  Total detections:  {total_detections}")
if confidences:
    print(f"  Conf min/avg/max:  {min(confidences):.2f} / {sum(confidences)/len(confidences):.2f} / {max(confidences):.2f}")
print(f"{'─'*45}")
if save_dir:
    print(f"  Saved to: {save_dir}")
