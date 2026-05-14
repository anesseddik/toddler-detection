"""
Full toddler pipeline: detection + tracking + pose estimation.

Flow per frame:
  1. Detection model + ByteTrack  ->  toddler bounding boxes with persistent IDs
  2. Pose model                   ->  all person skeletons on the full frame
  3. Matching                     ->  skeletons matched to tracked toddler boxes only
  4. Drawing                      ->  per-ID coloured box, ID label, skeleton, motion trail

Adults are ignored entirely. Every toddler gets a unique ID that persists across
the video, a motion trail, and a full body skeleton.

Usage:
    python inference/track_and_pose.py                    # webcam
    python inference/track_and_pose.py path/to/video.mp4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import timedelta
from ultralytics import YOLO
from config import FINAL_DETECT_MODEL, RUNS_DIR

SOURCE        = sys.argv[1] if len(sys.argv) > 1 else "0"
DETECT_CONF   = 0.4
POSE_CONF     = 0.4
KP_CONF_THR   = 0.4
TRAIL_LEN     = 50
EXIT_PATIENCE = 45
DISPLAY_WIDTH = 960

detect_model = YOLO(str(FINAL_DETECT_MODEL))
pose_model   = YOLO("yolov8n-pose.pt")

# ── COCO skeleton ──────────────────────────────────────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KP_COLOR = [
    (0, 230, 0),   (0, 230, 0),   (0, 230, 0),   (0, 230, 0),   (0, 230, 0),
    (255, 128, 0), (255, 128, 0),
    (0, 165, 255), (0, 165, 255),
    (0, 80,  255), (0, 80,  255),
    (0, 128, 255), (0, 128, 255),
    (0, 200, 255), (0, 200, 255),
    (0, 255, 200), (0, 255, 200),
]


def track_color(tid):
    hue = int((tid * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def center_in_box(pose_xyxy, det_xyxy, margin=30):
    cx = (pose_xyxy[0] + pose_xyxy[2]) / 2
    cy = (pose_xyxy[1] + pose_xyxy[3]) / 2
    return (det_xyxy[0] - margin <= cx <= det_xyxy[2] + margin and
            det_xyxy[1] - margin <= cy <= det_xyxy[3] + margin)


def draw_skeleton(frame, kpts_xy, kpts_conf):
    for a, b in SKELETON:
        if kpts_conf[a] < KP_CONF_THR or kpts_conf[b] < KP_CONF_THR:
            continue
        pt1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
        pt2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
        cv2.line(frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)
    for i, (xy, conf) in enumerate(zip(kpts_xy, kpts_conf)):
        if conf < KP_CONF_THR:
            continue
        cx, cy = int(xy[0]), int(xy[1])
        color  = KP_COLOR[i] if i < len(KP_COLOR) else (200, 200, 200)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)


def ts(frame_idx, fps):
    return str(timedelta(seconds=int(frame_idx / fps)))


# ── Video setup ────────────────────────────────────────────────────────────────
_src = int(SOURCE) if SOURCE.isdigit() else SOURCE
_cap = cv2.VideoCapture(_src)
fps    = _cap.get(cv2.CAP_PROP_FPS) or 25.0
width  = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_cap.release()

writer    = None
save_path = None
if not SOURCE.isdigit():
    out_dir   = RUNS_DIR / "track_pose"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / (Path(SOURCE).stem + "_tracked_pose.mp4")
    writer    = cv2.VideoWriter(
        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

# ── Per-track state ────────────────────────────────────────────────────────────
trails     = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
first_seen = {}
last_seen  = {}
active     = set()
all_ids    = set()
frame_num  = 0

print(f"\nDetection : {Path(FINAL_DETECT_MODEL).name}")
print(f"Tracker   : ByteTrack")
print(f"Pose      : yolov8n-pose.pt  (17 keypoints)")
print(f"Source    : {SOURCE}")
print(f"{'─' * 55}")

# ── Main loop ──────────────────────────────────────────────────────────────────
for r in detect_model.track(
    source=SOURCE,
    conf=DETECT_CONF,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    save=False,
    verbose=False,
):
    frame_num += 1
    frame = r.orig_img.copy()

    # ── Tracked toddler boxes ──────────────────────────────────────────────────
    ids_this_frame  = set()
    tracked_boxes   = []   # list of (tid, x1, y1, x2, y2)

    if r.boxes is not None and r.boxes.id is not None:
        ids_arr   = r.boxes.id.cpu().numpy().astype(int)
        boxes_arr = r.boxes.xyxy.cpu().numpy()

        for tid, box in zip(ids_arr, boxes_arr):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trails[tid].append((cx, cy))
            last_seen[tid] = frame_num
            ids_this_frame.add(tid)
            tracked_boxes.append((tid, x1, y1, x2, y2))

            if tid not in all_ids:
                all_ids.add(tid)
                first_seen[tid] = frame_num
                print(f"  [+] Toddler ID {tid:3d}  entered  @ {ts(frame_num, fps)}")

    # Exit events
    for tid in list(active):
        if tid not in ids_this_frame:
            if frame_num - last_seen.get(tid, frame_num) >= EXIT_PATIENCE:
                dur = last_seen[tid] - first_seen[tid]
                print(f"  [-] Toddler ID {tid:3d}  exited   @ {ts(last_seen[tid], fps)}"
                      f"  ({dur} frames)")
                active.discard(tid)
    active.update(ids_this_frame)

    # ── Pose estimation ────────────────────────────────────────────────────────
    pose_r = pose_model.predict(frame, conf=POSE_CONF, verbose=False)[0]

    # ── Draw trails ───────────────────────────────────────────────────────────
    for tid, trail in trails.items():
        pts   = list(trail)
        color = track_color(tid)
        for i in range(1, len(pts)):
            alpha     = i / len(pts)
            thickness = max(1, int(alpha * 3))
            faded     = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], faded, thickness)

    # ── Draw detection boxes with ID labels ───────────────────────────────────
    for tid, x1, y1, x2, y2 in tracked_boxes:
        color = track_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label  = f"Toddler #{tid}"
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Draw skeletons matched to tracked toddlers ────────────────────────────
    if (pose_r.keypoints is not None
            and pose_r.boxes is not None
            and len(pose_r.keypoints) > 0):

        pose_boxes = pose_r.boxes.xyxy.cpu().numpy()
        kpts_xy    = pose_r.keypoints.xy.cpu().numpy()
        kpts_conf  = pose_r.keypoints.conf.cpu().numpy()

        for pbox, kxy, kcf in zip(pose_boxes, kpts_xy, kpts_conf):
            det_boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in tracked_boxes]
            if any(center_in_box(pbox, db) for db in det_boxes):
                draw_skeleton(frame, kxy, kcf)

    # ── Display / save ─────────────────────────────────────────────────────────
    if writer:
        writer.write(frame)

    h, w  = frame.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.imshow("Toddler Track + Pose  [Q to quit]",
               cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if writer:
    writer.release()
cv2.destroyAllWindows()

# Final exits for still-active tracks
for tid in active:
    dur = last_seen.get(tid, frame_num) - first_seen.get(tid, 0)
    print(f"  [-] Toddler ID {tid:3d}  exited   @ {ts(last_seen.get(tid, frame_num), fps)}"
          f"  ({dur} frames)")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─' * 55}")
print(f"  Source           : {SOURCE}")
print(f"  Total frames     : {frame_num}")
print(f"  Unique toddlers  : {len(all_ids)}")
if all_ids:
    durations = {tid: last_seen.get(tid, frame_num) - first_seen.get(tid, 0)
                 for tid in all_ids}
    longest   = max(durations, key=durations.get)
    print(f"  Longest track    : ID {longest}  ({durations[longest]} frames)")
print(f"{'─' * 55}")
if save_path:
    print(f"  Saved to: {save_path}")
