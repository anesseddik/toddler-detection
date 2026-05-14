# YOLOv8 Toddler Detection — Project Progress

## Accomplished

### Detection Model
- Fine-tuned YOLOv8s on a custom toddler detection dataset (single class)
- 5 training runs total, each improving on the last
- **Final model (Run 5):** `models/toddler_detect_s100_best.pt`
  - Test mAP50: **0.959**
  - Test mAP50-95: **0.575**
  - Trained on Google Colab A100, 100 epochs

### Training Runs History
| Run | Description | Val mAP50 | Test mAP50 | Test mAP50-95 |
|-----|-------------|-----------|------------|---------------|
| 1 | YOLOv8n, naive split | 0.948 | 0.700 | 0.310 |
| 2 | YOLOv8s, proper 70/15/15 re-split | 0.956 | 0.936 | 0.537 |
| 3 | Run 2 + Roboflow adult hard negatives | 0.971 | 0.951 | 0.565 |
| 4 | Run 3 + custom video hard negatives | 0.961 | 0.951 | 0.567 |
| **5** | **Run 4 + pets/furniture/rooms/toys hard negatives** | **—** | **0.959** | **0.575** |

### Hard Negatives (all added to training set)
- Roboflow "Children vs Adults" dataset — adult-only images (cell 6b)
- 74 custom adult video frames extracted locally (cell 6c)
- Oxford IIIT pets — cats & dogs (cell 6d)
- Furniture detection dataset — chairs, tables, sofas (cell 6d)
- Indoor room scenes — walls, floors, windows (cell 6d)
- Toys dataset (cell 6d)

### False Positive Testing (Run 5)
Tested on 5 adult-only videos recorded locally:

| Video | Frames | FP Frames | FP% | Max Conf | Result |
|-------|--------|-----------|-----|----------|--------|
| video_2026-05-13_19-29-50.mp4 | 1207 | 3 | 0.2% | 0.56 | ⚠ Low |
| IMG_3638.mp4 | 1888 | 13 | 0.7% | 0.84 | ⚠ Investigate |
| IMG_1127.mp4 | 977 | 2 | 0.2% | 0.69 | ⚠ Low |
| IMG_3647.mp4 | 1606 | 0 | 0.0% | — | ✓ Clean |
| IMG_3648.MP4 | 1822 | 0 | 0.0% | — | ✓ Clean |

IMG_3638 has 13 false positives with max conf 0.84 — annotated video saved at `runs/test_results/`.

### Infrastructure
- `config.py` — central path configuration
- `training/train_detect.py` — local CPU training script
- `training/train_pose.py` — pose training script (ready, dataset not downloaded)
- `inference/detect.py` — video/image/webcam inference (conf=0.4)
- `inference/test_videos.py` — batch false positive tester
- `inference/generate_report.py` — PDF report generator
- `notebooks/train_colab.ipynb` — full Colab training pipeline (cells 1–9, includes cell 6d)
- `data/download_roboflow_negatives.py` — downloads hard negatives from Roboflow locally

---

## Current State

| Component | Status |
|-----------|--------|
| Detection model | Done |
| Hard negatives | Done |
| Inference script | Done |
| PDF report | Needs update (still shows Run 4 results) |
| Pose model | Not started |
| Combined pipeline | Not started |

---

## Next Steps

1. **Investigate IMG_3638 false positives** — open `runs/test_results/IMG_3638/` annotated video, identify what triggered conf 0.84, consider adding more targeted hard negatives if needed
2. **Pose model** — download SyRIP dataset (`data/download_syrip.py`), convert to YOLO format (`data/convert_syrip_to_yolo.py`), train with `training/train_pose.py`
3. **Combined detection + pose pipeline** — fuse both models: detection finds the toddler, pose estimates keypoints for danger detection (e.g. near stairs, on furniture, reaching upward)
4. **Update PDF report** — update `inference/generate_report.py` to reflect Run 5 results and new hard negative categories
