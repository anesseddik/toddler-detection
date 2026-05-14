import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from config import ADULT_NEG_DIR

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else input("Enter video path: ")
OUTPUT_DIR = ADULT_NEG_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SEC = 2  # one frame every 2 seconds

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    sys.exit(1)

fps             = cap.get(cv2.CAP_PROP_FPS)
interval_frames = max(1, int(fps * INTERVAL_SEC))
total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.1f} fps, {total_frames} frames (~{total_frames/fps:.0f}s)")
print(f"Extracting 1 frame every {INTERVAL_SEC}s...")

# Continue from highest existing index so multiple runs don't overwrite each other
existing = list(OUTPUT_DIR.glob("adult_neg_*.jpg"))
start_idx = (max(int(p.stem.split("_")[-1]) for p in existing) + 1) if existing else 0
print(f"Existing frames in folder: {len(existing)}  (starting from index {start_idx})")

frame_idx = 0
saved     = start_idx
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % interval_frames == 0:
        cv2.imwrite(str(OUTPUT_DIR / f"adult_neg_{saved:04d}.jpg"), frame)
        saved += 1
    frame_idx += 1

cap.release()
newly_saved = saved - start_idx
print(f"\nExtracted {newly_saved} new frames  (total in folder: {saved})")
print(f"Saved to: {OUTPUT_DIR}")
print("NEXT STEP: Open that folder and DELETE any frames that contain a toddler.")
print("Keep only frames where ONLY adults are visible.")
