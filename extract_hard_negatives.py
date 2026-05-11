import cv2
import sys
from pathlib import Path

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else input("Enter video path: ")
OUTPUT_DIR = Path("C:/pfe/adult_negatives")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SEC = 2  # one frame every 2 seconds

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = max(1, int(fps * INTERVAL_SEC))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.1f} fps, {total_frames} frames (~{total_frames/fps:.0f}s)")
print(f"Extracting 1 frame every {INTERVAL_SEC}s...")

frame_idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % interval_frames == 0:
        cv2.imwrite(str(OUTPUT_DIR / f"adult_neg_{saved:04d}.jpg"), frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"\nExtracted {saved} frames to {OUTPUT_DIR}")
print("NEXT STEP: Open that folder and DELETE any frames that contain a toddler.")
print("Keep only frames where ONLY adults are visible.")
