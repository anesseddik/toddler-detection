"""
Toddler body-part detection using the pre-trained YOLOv8n-pose model.
No fine-tuning needed — COCO-pretrained pose works on people of all ages.

Detects 17 keypoints per person:
  0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
  5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
  9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
  13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

Usage:
    python inference/pose.py                    # webcam
    python inference/pose.py path/to/video.mp4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from ultralytics import YOLO

SOURCE        = sys.argv[1] if len(sys.argv) > 1 else "0"
CONF          = 0.4
DISPLAY_WIDTH = 960
MODEL         = "yolov8n-pose.pt"   # auto-downloaded on first run (~6 MB)

model = YOLO(MODEL)

print(f"\nModel  : {MODEL}  (17 COCO keypoints)")
print(f"Source : {SOURCE}")
print(f"Conf   : {CONF}")
print("─" * 45)

total_frames   = 0
frames_with_kp = 0
save_dir       = None

for r in model.predict(
    source=SOURCE,
    conf=CONF,
    save=True,
    stream=True,
    verbose=False,
):
    total_frames += 1
    if r.keypoints is not None and len(r.keypoints) > 0:
        frames_with_kp += 1

    if save_dir is None:
        save_dir = r.save_dir

    frame = r.plot()
    h, w  = frame.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.imshow("Toddler Pose  [Q to quit]",
               cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print(f"\n  Total frames     : {total_frames}")
print(f"  Frames with pose : {frames_with_kp}  "
      f"({100 * frames_with_kp / max(total_frames, 1):.1f}%)")
print("─" * 45)
if save_dir:
    print(f"  Saved to: {save_dir}")
