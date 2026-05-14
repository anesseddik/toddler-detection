"""
Combined toddler detection + pose estimation pipeline.

Flow per frame:
  1. Detection model  →  finds toddler bounding boxes
  2. Pose model       →  finds all person skeletons on the full frame
  3. Matching         →  keeps only skeletons whose center falls inside a toddler box
  4. Drawing          →  detection box + skeleton drawn only on confirmed toddlers

Result: adults are ignored completely; every toddler gets a labelled box and skeleton.

Usage:
    python inference/detect_and_pose.py                    # webcam
    python inference/detect_and_pose.py path/to/video.mp4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ultralytics import YOLO
from config import FINAL_DETECT_MODEL, RUNS_DIR

SOURCE        = sys.argv[1] if len(sys.argv) > 1 else "0"
DETECT_CONF   = 0.4
POSE_CONF     = 0.4
DISPLAY_WIDTH = 960
KP_CONF_THR   = 0.4   # minimum keypoint confidence to draw

detect_model = YOLO(str(FINAL_DETECT_MODEL))
pose_model   = YOLO("yolov8n-pose.pt")

# ── COCO skeleton connections (joint index pairs) ──────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6),                                     # shoulders bar
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12),                           # torso sides
    (11, 12),                                   # hips bar
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]

# BGR colors per keypoint (face=green, shoulders/arms=blue, hips/legs=orange)
KP_COLOR = [
    (0, 230, 0),   (0, 230, 0),   (0, 230, 0),   (0, 230, 0),   (0, 230, 0),
    (255, 128, 0), (255, 128, 0),
    (0, 165, 255), (0, 165, 255),
    (0, 80, 255),  (0, 80, 255),
    (0, 128, 255), (0, 128, 255),
    (0, 200, 255), (0, 200, 255),
    (0, 255, 200), (0, 255, 200),
]

LIMB_COLOR = (180, 180, 180)   # skeleton lines


def center_in_box(pose_xyxy, det_xyxy, margin=30):
    """True if the center of pose_xyxy is inside det_xyxy (with pixel margin)."""
    cx = (pose_xyxy[0] + pose_xyxy[2]) / 2
    cy = (pose_xyxy[1] + pose_xyxy[3]) / 2
    return (det_xyxy[0] - margin <= cx <= det_xyxy[2] + margin and
            det_xyxy[1] - margin <= cy <= det_xyxy[3] + margin)


def draw_skeleton(frame, kpts_xy, kpts_conf):
    """Draw keypoints and limb connections on frame in-place."""
    n = len(kpts_xy)

    # Limb connections
    for a, b in SKELETON:
        if a >= n or b >= n:
            continue
        if kpts_conf[a] < KP_CONF_THR or kpts_conf[b] < KP_CONF_THR:
            continue
        pt1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
        pt2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
        cv2.line(frame, pt1, pt2, LIMB_COLOR, 2, cv2.LINE_AA)

    # Keypoint dots
    for i, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
        if conf < KP_CONF_THR:
            continue
        cx, cy = int(xy[0]), int(xy[1])
        color  = KP_COLOR[i] if i < len(KP_COLOR) else (200, 200, 200)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)


# ── Video source setup ─────────────────────────────────────────────────────────
src     = int(SOURCE) if SOURCE.isdigit() else SOURCE
cap     = cv2.VideoCapture(src)
fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

save_path = None
writer    = None
if not SOURCE.isdigit():
    out_dir   = RUNS_DIR / "detect_pose"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / (Path(SOURCE).stem + "_pose.mp4")
    writer    = cv2.VideoWriter(
        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

print(f"\nDetection model : {Path(FINAL_DETECT_MODEL).name}")
print(f"Pose model      : yolov8n-pose.pt  (17 COCO keypoints)")
print(f"Source          : {SOURCE}")
print(f"─" * 55)

total_frames = 0
frames_with_toddler = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    total_frames += 1

    # ── 1. Detection ───────────────────────────────────────────────────────────
    det_results  = detect_model.predict(frame, conf=DETECT_CONF, verbose=False)[0]
    toddler_boxes = (det_results.boxes.xyxy.cpu().numpy()
                     if det_results.boxes is not None and len(det_results.boxes) > 0
                     else [])

    # ── 2. Pose ────────────────────────────────────────────────────────────────
    pose_results = pose_model.predict(frame, conf=POSE_CONF, verbose=False)[0]

    # ── 3. Draw detection boxes ────────────────────────────────────────────────
    annotated = frame.copy()
    for box in toddler_boxes:
        frames_with_toddler += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 90, y1), (0, 200, 0), -1)
        cv2.putText(annotated, "Toddler", (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 4. Match pose to toddler boxes & draw skeleton ─────────────────────────
    if (pose_results.keypoints is not None
            and pose_results.boxes is not None
            and len(pose_results.keypoints) > 0):

        pose_boxes_np = pose_results.boxes.xyxy.cpu().numpy()
        kpts_xy_all   = pose_results.keypoints.xy.cpu().numpy()    # (M, 17, 2)
        kpts_cf_all   = pose_results.keypoints.conf.cpu().numpy()  # (M, 17)

        for pbox, kpts_xy, kpts_conf in zip(pose_boxes_np, kpts_xy_all, kpts_cf_all):
            matched = any(center_in_box(pbox, tbox) for tbox in toddler_boxes)
            if matched:
                draw_skeleton(annotated, kpts_xy, kpts_conf)

    # ── 5. Display / save ──────────────────────────────────────────────────────
    if writer:
        writer.write(annotated)

    h, w  = annotated.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.imshow("Toddler Detect + Pose  [Q to quit]",
               cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n  Total frames         : {total_frames}")
print(f"  Frames with toddler  : {frames_with_toddler}  "
      f"({100 * frames_with_toddler / max(total_frames, 1):.1f}%)")
print(f"─" * 55)
if save_path:
    print(f"  Saved to: {save_path}")
