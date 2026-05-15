# YOLOv8 Toddler Detection & Monitoring — Project Progress

---

## Accomplished

### 1. Detection Model
- Fine-tuned YOLOv8s on a custom toddler detection dataset (single class: toddler)
- 5 training runs, each improving on the last
- **Final model (Run 5):** `models/toddler_detect_s100_best.pt` — best.pt = epoch 89 (peak val mAP50)
  - Trained on Google Colab A100, 100 epochs, batch=64

| Metric | Value |
|--------|-------|
| Precision | **94.47%** |
| Recall | **93.30%** |
| F1 Score | **93.88%** |
| mAP50 (val, epoch 89) | **96.53%** |
| mAP50 (test) | **95.90%** |
| mAP50-95 (test) | **57.50%** |

| Run | Description | Val mAP50 | Test mAP50 | Test mAP50-95 |
|-----|-------------|-----------|------------|---------------|
| 1 | YOLOv8n, naive split | 0.948 | 0.700 | 0.310 |
| 2 | YOLOv8s, proper 70/15/15 re-split | 0.956 | 0.936 | 0.537 |
| 3 | Run 2 + Roboflow adult hard negatives | 0.971 | 0.951 | 0.565 |
| 4 | Run 3 + custom video hard negatives (74 frames) | 0.961 | 0.951 | 0.567 |
| **5** | **Run 4 + pets/furniture/rooms/toys hard negatives** | **—** | **0.959** | **0.575** |

### 2. Hard Negative Mining (4 stages)
- Roboflow "Children vs Adults" — adult-only images
- 74 custom adult video frames extracted locally
- Oxford IIIT Pets — cats & dogs
- Furniture detection — chairs, tables, sofas
- Indoor room scenes — walls, floors, windows
- Toys detection dataset

### 3. False Positive Testing (Run 5)
Tested on 5 adult-only videos (7,500 total frames). Overall FP rate: **0.24%**

| Video | Frames | FP Frames | FP% | Max Conf | Result |
|-------|--------|-----------|-----|----------|--------|
| video_2026-05-13_19-29-50.mp4 | 1207 | 3 | 0.2% | 0.56 | ⚠ Low |
| IMG_3638.mp4 | 1888 | 13 | 0.7% | 0.84 | ⚠ Investigate |
| IMG_1127.mp4 | 977 | 2 | 0.2% | 0.69 | ⚠ Low |
| IMG_3647.mp4 | 1606 | 0 | 0.0% | — | ✓ Clean |
| IMG_3648.MP4 | 1822 | 0 | 0.0% | — | ✓ Clean |

IMG_3638 has 13 FP frames (max conf 0.84) — annotated video at `runs/test_results/IMG_3638/`.

### 4. Multi-Object Tracking
- Added ByteTrack (built into Ultralytics) — no extra dependencies
- Each toddler gets a persistent ID across the full video
- Motion trail drawn behind each toddler (last 50 positions, fades with age)
- Entry/exit events logged to console with timestamps
- Handles occlusions: IDs recovered after two toddlers briefly overlap
- Script: `inference/track.py`

#### 4a. ByteTrack Tuning — ID Fragmentation Fix (2026-05-15)
Initial test showed ID fragmentation (31 IDs in 4 s). Root cause: detections dropping below
conf threshold for 1–2 frames caused ByteTrack to lose and reassign tracks.

Fix — decoupled detection floor from track-creation threshold (`bytetrack_toddler.yaml`):

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `DETECT_CONF` (script) | 0.4 | 0.1 | Passes weak detections to tracker for recovery |
| `track_high_thresh` | 0.25 | 0.4 | Only strong detections drive Pass 1 matching |
| `track_low_thresh` | 0.1 | 0.05 | Weak detections used only for Pass 2 recovery |
| `new_track_thresh` | 0.25 | 0.5 | Requires high confidence to start a new track |
| `track_buffer` | 30 (1 s) | 90 (3 s) | Tracks survive longer occlusions |

Result on `multi_toddler tracking test.mp4`:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Frames processed | 123 | 215 |
| Unique IDs | 31 | 38 |
| Longest track | 121 frames (4 s) | **214 frames (7.1 s)** |
| IDs / second | 7.8 | **5.3** |

### 5. Pose Estimation (Body Part Detection)
- YOLOv8n-pose (COCO pretrained) detects 17 keypoints on each toddler
- No fine-tuning required — generalises well to toddlers out of the box
- Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- Script: `inference/pose.py`

### 6. Combined Pipeline
- `inference/detect_and_pose.py` — detection + pose, adults filtered out
- `inference/track_and_pose.py` — full pipeline: detection + ByteTrack + pose
  - Detection model acts as gatekeeper: only confirmed toddlers get a skeleton
  - Per-ID coloured box, persistent ID label, motion trail, and skeleton all drawn simultaneously
  - Annotated video saved to `runs/track_pose/`

### 7. Reports
- `inference/generate_report.py` — PDF report generator (fpdf2), covers all 14 sections
- `inference/generate_report_word.py` — Word report generator (python-docx), same content
- Generated at `reports/toddler_detection_report.pdf` and `reports/toddler_detection_report.docx`
- Sections: overview, tools & technologies (detailed), dataset, training config, 5-run history,
  final results (precision/recall/F1 + best.pt explanation), FP testing, problems/solutions,
  tracking (ByteTrack tuning), pose, combined pipeline, project structure, next steps,
  training plots (15 figures: loss curves, confusion matrices, P/R/F1/PR curves, train & val batches)

---

## Current State

| Component | Status |
|-----------|--------|
| Detection model (Run 5) | Done |
| Hard negatives (4 categories) | Done |
| False positive testing | Done |
| Multi-object tracking | Done |
| ByteTrack tuning (ID fragmentation fix) | Done |
| Pose estimation (body parts) | Done |
| Combined detection + tracking + pose | Done |
| PDF report | Done (up to date) |
| Word report | Done (up to date) |
| Danger detection logic | Not started |
| Alert / notification system | Not started |

---

## Project Structure

```
inference/
  detect.py                — detection only
  track.py                 — detection + tracking
  pose.py                  — detection + pose
  detect_and_pose.py       — detection + pose (adults filtered)
  track_and_pose.py        — full pipeline (detection + tracking + pose)
  test_videos.py           — batch FP tester
  generate_report.py       — PDF report generator
  generate_report_word.py  — Word report generator

bytetrack_toddler.yaml     — tuned ByteTrack config (track_buffer=90, new_track_thresh=0.5)

training/
  train_detect.py       — local detection training (CPU-safe)
  train_pose.py         — pose fine-tuning script (ready if needed)

data/
  prepare_dataset.py              — 70/15/15 re-split utility
  extract_hard_negatives.py       — extracts frames from video
  download_roboflow_negatives.py  — downloads Roboflow hard negatives
  download_syrip.py               — SyRIP dataset downloader (needs Drive IDs)
  convert_syrip_to_yolo.py        — converts SyRIP to YOLO pose format

notebooks/
  train_colab.ipynb     — full Colab training pipeline (cells 1-9 + 6d)

models/
  toddler_detect_s100_best.pt    — final detection model (21.5 MB)

reports/
  toddler_detection_report.pdf   — PDF report (14 sections, fpdf2)
  toddler_detection_report.docx  — Word report (same content, python-docx)

runs/
  track_pose/                    — annotated output from full pipeline runs
  test_results/                  — annotated output from FP testing (incl. IMG_3638/)
```

---

## Next Steps

1. **Danger detection logic** — use keypoints from `track_and_pose.py` to flag dangerous postures:
   - Wrists above nose → reaching or climbing
   - Ankle Y near hip Y → fallen or crawling
   - Large sudden centre-of-mass jump → rapid fall
   - Toddler near edge of frame → leaving monitored zone

2. **Alert system** — trigger a sound or desktop notification when a danger condition persists for N consecutive frames (to avoid false alarms from single-frame noise)

3. **Investigate IMG_3638 false positives** — open annotated video at `runs/test_results/IMG_3638/`, identify what triggered conf 0.84, add targeted hard negatives if needed

4. **Pose fine-tuning (optional)** — if keypoint accuracy is insufficient for danger detection on specific camera angles, fine-tune `yolov8n-pose.pt` using `training/train_pose.py` with a toddler-specific dataset
