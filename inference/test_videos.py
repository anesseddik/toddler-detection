"""
Run inference on a list of test videos and print a false-positive summary.
No display window — results are saved to runs/test_results/.

Usage:
    python inference/test_videos.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from config import FINAL_DETECT_MODEL

VIDEOS = [
    r"C:\Users\MrCOMPUTER\Downloads\Telegram Desktop\video_2026-05-13_19-29-50.mp4",
    r"C:\Users\MrCOMPUTER\Downloads\Telegram Desktop\IMG_3638.mp4",
    r"C:\Users\MrCOMPUTER\Downloads\Telegram Desktop\IMG_1127.mp4",
    r"C:\Users\MrCOMPUTER\Downloads\Telegram Desktop\IMG_3647.mp4",
    r"C:\Users\MrCOMPUTER\Downloads\Telegram Desktop\IMG_3648.MP4",
]

CONF      = 0.4
SAVE_DIR  = Path(__file__).resolve().parents[1] / "runs" / "test_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(FINAL_DETECT_MODEL))

print(f"\nModel : {FINAL_DETECT_MODEL.name}")
print(f"Conf  : {CONF}")
print(f"Output: {SAVE_DIR}\n")
print(f"{'Video':<45} {'Frames':>7} {'Det.Frames':>10} {'FP%':>6} {'MaxConf':>8}")
print("─" * 80)

for video_path in VIDEOS:
    p = Path(video_path)
    if not p.exists():
        print(f"{p.name:<45} {'NOT FOUND':>7}")
        continue

    total_frames = 0
    frames_with_det = 0
    max_conf = 0.0

    for r in model.predict(
        source=str(p),
        conf=CONF,
        save=True,
        save_dir=str(SAVE_DIR / p.stem),
        stream=True,
        verbose=False,
    ):
        total_frames += 1
        if len(r.boxes) > 0:
            frames_with_det += 1
            c = float(r.boxes.conf.max())
            if c > max_conf:
                max_conf = c

    fp_pct = 100 * frames_with_det / max(total_frames, 1)
    max_conf_str = f"{max_conf:.2f}" if frames_with_det > 0 else "—"
    flag = " ⚠" if frames_with_det > 0 else " ✓"
    print(f"{p.name:<45} {total_frames:>7} {frames_with_det:>10} {fp_pct:>5.1f}% {max_conf_str:>8}{flag}")

print("─" * 80)
print(f"\nAnnotated videos saved to: {SAVE_DIR}")
print("✓ = no false positives   ⚠ = false positives detected")
