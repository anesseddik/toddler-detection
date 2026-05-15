"""
Multi-object toddler tracking using ByteTrack (built into Ultralytics).
Each toddler gets a persistent ID across frames. Trails are drawn to show
movement paths. Entry / exit events are printed with timestamps.

Usage:
    python inference/track.py                  # webcam
    python inference/track.py path/to/video.mp4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import timedelta
from ultralytics import YOLO
from config import FINAL_DETECT_MODEL

SOURCE        = sys.argv[1] if len(sys.argv) > 1 else "0"
CONF          = 0.1   # low floor so tracker receives weak detections for track recovery
TRAIL_LEN     = 50    # number of past positions to draw per track
EXIT_PATIENCE = 90    # frames absent before an exit event is logged (matches track_buffer)
DISPLAY_WIDTH = 960
TRACKER_CFG   = str(Path(__file__).resolve().parents[1] / "bytetrack_toddler.yaml")

model = YOLO(str(FINAL_DETECT_MODEL))

# ── Get source FPS for human-readable timestamps ───────────────────────────────
_src = int(SOURCE) if SOURCE.isdigit() else SOURCE
_cap = cv2.VideoCapture(_src)
fps  = _cap.get(cv2.CAP_PROP_FPS) or 25.0
_cap.release()

def ts(frame_idx):
    return str(timedelta(seconds=int(frame_idx / fps)))

def track_color(tid):
    hue = int((tid * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

# ── Per-track state ────────────────────────────────────────────────────────────
trails      = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
first_seen  = {}   # tid -> frame number of first appearance
last_seen   = {}   # tid -> frame number of last detection
active      = set()
all_ids     = set()
frame_num   = 0
save_dir    = None

print(f"\nTracker : ByteTrack  |  Conf: {CONF}  |  Source: {SOURCE}")
print(f"FPS     : {fps:.1f}")
print(f"{'─' * 55}")

# ── Main tracking loop ─────────────────────────────────────────────────────────
for r in model.track(
    source=SOURCE,
    conf=CONF,
    tracker=TRACKER_CFG,
    persist=True,
    stream=True,
    save=True,
    line_width=2,
    verbose=False,
):
    frame_num += 1
    if save_dir is None:
        save_dir = r.save_dir

    ids_this_frame = set()

    if r.boxes.id is not None:
        ids_arr   = r.boxes.id.cpu().numpy().astype(int)
        boxes_arr = r.boxes.xyxy.cpu().numpy()

        for tid, box in zip(ids_arr, boxes_arr):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            trails[tid].append((cx, cy))
            last_seen[tid] = frame_num
            ids_this_frame.add(tid)

            if tid not in all_ids:
                all_ids.add(tid)
                first_seen[tid] = frame_num
                print(f"  [+] Toddler ID {tid:3d}  entered  @ {ts(frame_num)}")

    # Detect exits (patience-based so brief occlusions don't trigger false exits)
    for tid in list(active):
        if tid not in ids_this_frame:
            absent = frame_num - last_seen.get(tid, frame_num)
            if absent >= EXIT_PATIENCE:
                dur = last_seen[tid] - first_seen[tid]
                print(f"  [-] Toddler ID {tid:3d}  exited   @ {ts(last_seen[tid])}  "
                      f"({dur} frames tracked)")
                active.discard(tid)

    active.update(ids_this_frame)

    # Draw motion trails on top of the annotated frame
    frame = r.plot()
    for tid, trail in trails.items():
        pts = list(trail)
        color = track_color(tid)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(alpha * 3))
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], faded, thickness)

    h, w  = frame.shape[:2]
    scale = DISPLAY_WIDTH / w
    cv2.imshow("Toddler Tracker  [Q to quit]",
               cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# Log any tracks still active when the video ends
for tid in active:
    dur = last_seen.get(tid, frame_num) - first_seen.get(tid, 0)
    print(f"  [-] Toddler ID {tid:3d}  exited   @ {ts(last_seen.get(tid, frame_num))}  "
          f"({dur} frames tracked)")

# ── Session summary ────────────────────────────────────────────────────────────
print(f"\n{'─' * 55}")
print(f"  Source           : {SOURCE}")
print(f"  Total frames     : {frame_num}")
print(f"  Unique toddlers  : {len(all_ids)}")

if all_ids:
    durations  = {tid: last_seen.get(tid, frame_num) - first_seen.get(tid, 0)
                  for tid in all_ids}
    longest    = max(durations, key=durations.get)
    avg_dur    = sum(durations.values()) / len(durations)
    print(f"  Longest track    : ID {longest}  ({durations[longest]} frames / "
          f"{ts(durations[longest])})")
    print(f"  Avg track length : {avg_dur:.0f} frames")

print(f"{'─' * 55}")
if save_dir:
    print(f"  Saved to: {save_dir}")
