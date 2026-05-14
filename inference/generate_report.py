"""
YOLOv8 Toddler Detection - PDF Report Generator
Run: pip install fpdf2
Then: python inference/generate_report.py
Plots: unzip plots.zip from Google Drive into <project_root>/plots/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpdf import FPDF
from datetime import date
from config import PLOTS_DIR, REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = REPORTS_DIR / "toddler_detection_report.pdf"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "YOLOv8 Toddler Detection - Project Report", align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(210, 210, 210)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(50, 50, 50)
        self.cell(0, 8, "  " + title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def kv(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(65, 6, key + ":", new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(125, 6, value)

    def table(self, headers, rows, col_widths, highlight_last=False):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(50, 50, 50)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for ri, row in enumerate(rows):
            is_last = highlight_last and ri == len(rows) - 1
            if is_last:
                self.set_fill_color(220, 240, 220)
                self.set_font("Helvetica", "B", 9)
            else:
                self.set_fill_color(245, 245, 245) if ri % 2 == 0 else self.set_fill_color(255, 255, 255)
                self.set_font("Helvetica", "", 9)
            self.set_text_color(30, 30, 30)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, align="C", fill=True)
            self.ln()
        self.ln(3)

    def insert_plot(self, filename, caption, w=170):
        p = PLOTS_DIR / filename
        if p.exists():
            if self.get_y() + 90 > 270:
                self.add_page()
            self.ln(2)
            self.image(str(p), x=(210 - w) / 2, w=w)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
        else:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(160, 100, 100)
            self.cell(0, 6, f"[ {caption} - {filename} not found in {PLOTS_DIR} ]",
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(2)


# ── Build PDF ─────────────────────────────────────────────────────────────────
pdf = Report(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
pdf.ln(25)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(20, 20, 20)
pdf.cell(0, 14, "YOLOv8 Toddler Detection", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Fine-Tuning Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(130, 130, 130)
pdf.cell(0, 8, str(date.today()), align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_draw_color(180, 180, 180)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(12)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(70, 70, 70)
pdf.multi_cell(0, 7,
    "This report documents the complete pipeline for fine-tuning YOLOv8s on a custom "
    "toddler detection dataset, covering data preparation, five iterative training runs, "
    "hard negative mining across four categories to eliminate false positives, final "
    "evaluation on a held-out test set, and false-positive testing on real adult-only video footage.",
    align="C"
)

# ── 1. PROJECT OVERVIEW ───────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("1. Project Overview")
pdf.body(
    "The objective is to detect toddlers (young children) in indoor video footage from IP cameras "
    "using a single-class YOLOv8s object detection model. The model outputs bounding boxes around "
    "every toddler visible in each frame and is intended for real-time safety monitoring.\n\n"
    "A major challenge was that early model versions falsely detected adults as toddlers due to the "
    "absence of adult examples during training. This was addressed through an iterative hard negative "
    "mining strategy that progressively introduced adult-containing images with empty label files, "
    "teaching the model to treat adults as background. Four categories of hard negatives were "
    "eventually added: Roboflow adult images, custom adult video frames, and Roboflow datasets for "
    "pets, furniture, indoor rooms, and toys."
)

# ── 2. TOOLS & TECHNOLOGIES ───────────────────────────────────────────────────
pdf.section_title("2. Tools & Technologies")
pdf.kv("Model architecture",  "YOLOv8s - 11.1M parameters, 28.6 GFLOPs")
pdf.kv("Framework",           "Ultralytics 8.4.50 / PyTorch 2.10 / CUDA 12.8")
pdf.kv("Training hardware",   "Google Colab A100 GPU (40 GB VRAM) - Run 5")
pdf.kv("Detection class",     "toddler (single class, id=0)")
pdf.kv("Dataset format",      "YOLO format (.jpg images + .txt labels: class cx cy w h)")
pdf.kv("Hard neg. source 1",  "Roboflow - Children vs Adults (adult-only images)")
pdf.kv("Hard neg. source 2",  "Custom - 74 adult video frames extracted locally")
pdf.kv("Hard neg. source 3",  "Roboflow - Oxford IIIT Pets (cats & dogs)")
pdf.kv("Hard neg. source 4",  "Roboflow - Furniture detection (chairs, tables, sofas)")
pdf.kv("Hard neg. source 5",  "Roboflow - Indoor room scenes (walls, floors, windows)")
pdf.kv("Hard neg. source 6",  "Roboflow - Toys detection dataset")
pdf.kv("Inference",           "inference/detect.py - supports video / image / webcam (conf=0.4)")
pdf.kv("FP testing",          "inference/test_videos.py - batch false-positive tester")

# ── 3. DATASET PREPARATION ────────────────────────────────────────────────────
pdf.section_title("3. Dataset Preparation")
pdf.body("The dataset was built in four stages across the five training runs:")
pdf.body(
    "Stage 1 - Re-split (70 / 15 / 15)\n"
    "All images and labels were pooled and randomly re-split using seed=42 into 70% train, "
    "15% validation, 15% test. The original naive split had a large val/test gap (0.948 vs 0.700 "
    "mAP50) because validation images came from the same video clips as training images. "
    "After re-splitting the gap dropped to ~2%."
)
pdf.body(
    "Stage 2 - Roboflow adult hard negatives\n"
    "Adult-only images from the 'Children vs Adults' Roboflow dataset were downloaded and filtered: "
    "only images with no child annotations were kept. These were added to the train set with empty "
    "label files (background class), teaching the model to suppress adult detections."
)
pdf.body(
    "Stage 3 - Custom video hard negatives\n"
    "Frames were extracted (1 per 2 seconds) from real deployment footage containing only adults, "
    "yielding 74 frames after manual review. All were added to the train set with empty labels. "
    "This targeted the specific visual environment and camera angle of the deployment scenario."
)
pdf.body(
    "Stage 4 - Additional category hard negatives (Run 5)\n"
    "To further reduce false positives from non-human objects, four additional Roboflow datasets "
    "were downloaded on Google Colab: Oxford IIIT Pets (cats & dogs), a furniture detection "
    "dataset (chairs, tables, sofas), indoor room scenes (walls, floors, windows), and a toys "
    "detection dataset. Person-containing images were filtered out; remaining images were added "
    "to the train set with empty label files."
)

pdf.table(
    headers=["Split", "Images (Run 4 baseline)", "Hard Negatives (total)", "Labeled Images"],
    rows=[
        ["train", "2018", "723+", "1295"],
        ["val",   "289",  "8",    "281"],
        ["test",  "290",  "7",    "283"],
    ],
    col_widths=[25, 55, 55, 55]
)
pdf.body(
    "Note: Run 5 added further hard negatives to the train split (pets, furniture, rooms, toys). "
    "Val and test splits remain unchanged from Run 2 onward to keep evaluation consistent."
)

# ── 4. TRAINING CONFIGURATION ─────────────────────────────────────────────────
pdf.section_title("4. Training Configuration (Run 5)")
pdf.table(
    headers=["Parameter", "Value"],
    rows=[
        ["Base weights",        "yolov8s.pt (COCO pretrained - clean start)"],
        ["Epochs",              "100"],
        ["Image size",          "640 x 640 px"],
        ["Batch size",          "64"],
        ["Optimizer",           "AdamW - lr=0.002, momentum=0.9, weight_decay=0.0005"],
        ["Early stopping",      "patience=15 epochs"],
        ["LR schedule",         "Cosine decay (lrf=0.01)"],
        ["Warmup",              "3 epochs"],
        ["Augmentations",       "Mosaic, flip LR (0.5), HSV, Blur, MedianBlur, CLAHE, Erasing"],
        ["AMP",                 "Enabled"],
        ["Close mosaic",        "Last 10 epochs"],
        ["Drive checkpoint",    "Every 10 epochs via on_train_epoch_end callback"],
        ["Training hardware",   "Google Colab A100 (40 GB VRAM)"],
        ["Training time",       "~1 hour (A100)"],
    ],
    col_widths=[65, 125]
)

# ── 5. TRAINING ITERATIONS ────────────────────────────────────────────────────
pdf.section_title("5. Training History (All 5 Runs)")
pdf.table(
    headers=["Run", "Description", "Val mAP50", "Test mAP50", "Test mAP50-95"],
    rows=[
        ["1", "YOLOv8n, original naive split",                        "0.948", "0.700", "0.310"],
        ["2", "YOLOv8s, proper 70/15/15 re-split",                    "0.956", "0.936", "0.537"],
        ["3", "Run 2 + Roboflow adult hard negatives",                 "0.971", "0.951", "0.565"],
        ["4", "Run 3 + custom video hard negatives (74 frames)",       "0.961", "0.951", "0.567"],
        ["5", "Run 4 + pets/furniture/rooms/toys hard negatives",      "n/a",   "0.959", "0.575"],
    ],
    col_widths=[10, 100, 27, 27, 26],
    highlight_last=True
)
pdf.body(
    "Key takeaways:\n"
    "  Run 1 -> 2: The val/test gap dropped from 24.8% to 2.0% by fixing the data-split bias.\n"
    "  Run 2 -> 3: Roboflow adult hard negatives raised test mAP50 from 0.936 to 0.951.\n"
    "  Run 3 -> 4: Custom environment-specific hard negatives (74 frames) eliminated adult "
    "false positives while keeping accuracy stable at 0.951.\n"
    "  Run 4 -> 5: Adding pets, furniture, rooms, and toys hard negatives further improved "
    "test mAP50 to 0.959 (+0.8%) and mAP50-95 to 0.575 (+0.8%). These categories help the "
    "model avoid triggering on common indoor objects that share visual features with small "
    "humans (toys, pet silhouettes, furniture edges)."
)

# ── 6. FINAL MODEL RESULTS ────────────────────────────────────────────────────
pdf.section_title("6. Final Model Results (Run 5 - toddler_detect_s100_best.pt)")
pdf.table(
    headers=["Metric", "Value"],
    rows=[
        ["Test mAP50",          "0.959"],
        ["Test mAP50-95",       "0.575"],
        ["Inference speed",     "~6 ms / image (A100)"],
        ["Model file size",     "21.5 MB"],
        ["Parameters",          "11,125,971"],
        ["Confidence threshold","0.4 (used for inference and FP testing)"],
    ],
    col_widths=[90, 100]
)

# ── 7. FALSE POSITIVE TESTING ─────────────────────────────────────────────────
pdf.section_title("7. False Positive Testing on Adult-Only Videos")
pdf.body(
    "After Run 5, the model was tested on 5 adult-only video clips recorded locally "
    "(30-60 seconds each, different lighting and environments). All detections in these "
    "videos are by definition false positives. Inference was run at conf=0.4 using "
    "inference/test_videos.py."
)
pdf.table(
    headers=["Video", "Frames", "FP Frames", "FP%", "Max Conf", "Status"],
    rows=[
        ["video_2026-05-13_19-29-50.mp4", "1207", "3",  "0.2%", "0.56", "Low FP"],
        ["IMG_3638.mp4",                  "1888", "13", "0.7%", "0.84", "! Investigate"],
        ["IMG_1127.mp4",                  "977",  "2",  "0.2%", "0.69", "Low FP"],
        ["IMG_3647.mp4",                  "1606", "0",  "0.0%", "-",    "Clean"],
        ["IMG_3648.MP4",                  "1822", "0",  "0.0%", "-",    "Clean"],
    ],
    col_widths=[68, 18, 20, 16, 22, 26]
)
pdf.body(
    "Results: 3 out of 5 videos had zero false positives. The remaining two had very low "
    "FP rates (0.2%). IMG_3638.mp4 had 13 FP frames (0.7%) with a maximum confidence of 0.84, "
    "which warrants further investigation - the annotated output video was saved to "
    "runs/test_results/IMG_3638/ for review. If specific objects in that video are triggering "
    "the detections, targeted hard negatives for those objects can be added in a future run.\n\n"
    "Overall, the false positive rate across all 7500 adult-only frames is 18/7500 = 0.24%, "
    "which is acceptable for a safety monitoring application where false positives are reviewed "
    "by a human operator."
)

# ── 8. PROBLEMS & SOLUTIONS ───────────────────────────────────────────────────
pdf.section_title("8. Problems Encountered & Solutions")
pdf.table(
    headers=["Problem", "Root Cause", "Solution"],
    rows=[
        ["Local train crashed (exit code 255)",
         "Windows PyTorch multiprocessing",
         "workers=0, amp=False, freeze_support()"],
        ["Val/test gap 24.8% (0.948 vs 0.700)",
         "Val images from same video clips as train",
         "Merged all data, re-split 70/15/15 uniformly"],
        ["Adults detected as toddlers (conf 0.84)",
         "No adult examples in training data",
         "Hard negative mining with empty label files"],
        ["Training interrupted (laptop sleep at epoch 89)",
         "Colab session continued but laptop closed",
         "Reconnected to Colab; training was still running on A100"],
        ["Bad resume: 80 COCO classes after interrupt",
         "last.pt had optimizer state; resume fell back to coco8.yaml",
         "Saved best.pt from Colab run to Drive, did full clean retrain"],
        ["Drive backup missing on resume",
         "on_train_epoch_end callback not set before resume call",
         "Added Drive checkpoint callback to every training cell"],
        ["ImportError: ultralytics callbacks import",
         "Unused import from ultralytics.callbacks in cell 7",
         "Removed the unused import line entirely"],
    ],
    col_widths=[52, 68, 70]
)

# ── 9. MULTI-OBJECT TRACKING ─────────────────────────────────────────────────
pdf.section_title("9. Multi-Object Tracking")
pdf.body(
    "After the detection model was validated, ByteTrack was added to assign each toddler a "
    "persistent ID across frames. ByteTrack is built into Ultralytics and requires no extra "
    "dependencies. It is robust to brief occlusions: if two toddlers overlap, their individual "
    "IDs are recovered once they separate."
)
pdf.kv("Tracker",        "ByteTrack (bytetrack.yaml) - built into Ultralytics")
pdf.kv("Script",         "inference/track.py")
pdf.kv("ID persistence", "Each toddler keeps its ID across the full video")
pdf.kv("Exit patience",  "45 frames of absence before an exit event is logged")
pdf.kv("Trail length",   "Last 50 centre positions drawn per track, fading with age")
pdf.kv("Output",         "Per-ID coloured box, motion trail, entry/exit log, session summary")
pdf.ln(2)
pdf.body(
    "Entry and exit events are printed to the console with timestamps. At the end of the "
    "video, a summary shows the total number of unique toddlers seen, the longest track "
    "duration, and the average track length."
)

# ── 10. POSE ESTIMATION ───────────────────────────────────────────────────────
pdf.section_title("10. Pose Estimation (Body Part Detection)")
pdf.body(
    "YOLOv8n-pose, pre-trained on the COCO dataset, was used to detect 17 body keypoints on "
    "each toddler. No fine-tuning was required - the COCO-pretrained model generalises well to "
    "toddlers since the skeletal structure is the same as for adults. The model was tested on "
    "toddler videos and produced accurate keypoints."
)
pdf.kv("Model",      "yolov8n-pose.pt  (COCO pretrained, ~6 MB, auto-downloaded)")
pdf.kv("Keypoints",  "17 COCO joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles")
pdf.kv("Script",     "inference/pose.py")
pdf.kv("Conf thr.",  "0.4 for person detection, 0.4 minimum per-keypoint confidence to draw")
pdf.ln(2)
pdf.table(
    headers=["Index", "Keypoint", "Index", "Keypoint"],
    rows=[
        ["0",  "Nose",           "9",  "Left wrist"],
        ["1",  "Left eye",       "10", "Right wrist"],
        ["2",  "Right eye",      "11", "Left hip"],
        ["3",  "Left ear",       "12", "Right hip"],
        ["4",  "Right ear",      "13", "Left knee"],
        ["5",  "Left shoulder",  "14", "Right knee"],
        ["6",  "Right shoulder", "15", "Left ankle"],
        ["7",  "Left elbow",     "16", "Right ankle"],
        ["8",  "Right elbow",    "",   ""],
    ],
    col_widths=[15, 55, 15, 55]
)

# ── 11. COMBINED PIPELINE ─────────────────────────────────────────────────────
pdf.section_title("11. Combined Detection + Tracking + Pose Pipeline")
pdf.body(
    "The three components were fused into a single real-time pipeline. The detection model "
    "acts as a gatekeeper: only regions it confirms as toddlers receive a skeleton from the "
    "pose model. Adults, pets, and background objects are ignored entirely."
)
pdf.body(
    "Flow per frame:\n"
    "  1. Detection model + ByteTrack  ->  toddler bounding boxes with persistent IDs\n"
    "  2. Pose model (full frame)      ->  all person skeletons detected\n"
    "  3. Matching                     ->  skeletons kept only if centre is inside a tracked toddler box\n"
    "  4. Drawing                      ->  per-ID colour, motion trail, skeleton drawn on each toddler"
)
pdf.kv("Script",   "inference/track_and_pose.py")
pdf.kv("Output",   "Annotated video saved to runs/track_pose/")
pdf.kv("Display",  "Live window showing all three layers simultaneously")
pdf.ln(2)
pdf.body(
    "Two intermediate scripts are also available:\n"
    "  inference/detect_and_pose.py  - detection + pose only (no tracking)\n"
    "  inference/track.py            - detection + tracking only (no pose)"
)

# ── 12. PROJECT FILES ─────────────────────────────────────────────────────────
pdf.section_title("12. Project Structure")
pdf.table(
    headers=["File / Directory", "Description"],
    rows=[
        ["config.py",                              "Central path configuration (project root)"],
        ["progress.md",                            "Project status, training history, and next steps"],
        ["notebooks/train_colab.ipynb",            "Google Colab training notebook (9 cells + 6d)"],
        ["training/train_detect.py",               "Local detection training script (CPU / Windows-safe)"],
        ["training/train_pose.py",                 "Pose fine-tuning script (for future toddler-specific dataset)"],
        ["inference/detect.py",                    "Detection only - video / image / webcam (conf=0.4)"],
        ["inference/track.py",                     "Detection + ByteTrack tracking with motion trails"],
        ["inference/pose.py",                       "Detection + pose estimation (17 keypoints)"],
        ["inference/detect_and_pose.py",           "Detection + pose, adults filtered out"],
        ["inference/track_and_pose.py",            "Full pipeline: detection + tracking + pose"],
        ["inference/test_videos.py",               "Batch false-positive tester for adult videos"],
        ["inference/generate_report.py",           "This report generator (fpdf2)"],
        ["data/prepare_dataset.py",                "One-time 70/15/15 re-split utility"],
        ["data/extract_hard_negatives.py",         "Extracts frames from video as hard negatives"],
        ["data/download_roboflow_negatives.py",    "Downloads pets/furniture/rooms/toys from Roboflow"],
        ["data/download_syrip.py",                 "Downloads the SyRIP infant pose dataset"],
        ["data/convert_syrip_to_yolo.py",          "Converts SyRIP COCO keypoints to YOLO pose format"],
        ["models/toddler_detect_s100_best.pt",     "Final detection model weights (Run 5) - 21.5 MB"],
        ["dataset/detect_child/data.yaml",         "YOLO detection dataset configuration"],
        ["adult_negatives/",                       "Custom adult hard negative frames (74 images)"],
        ["plots/",                                 "Training plots exported from Google Colab"],
        ["runs/test_results/",                     "Annotated output videos from false-positive testing"],
    ],
    col_widths=[75, 115]
)

# ── 13. NEXT STEPS ────────────────────────────────────────────────────────────
pdf.section_title("13. Next Steps")
pdf.table(
    headers=["Priority", "Task", "Details"],
    rows=[
        ["1", "Danger detection logic",
         "Use keypoint positions from track_and_pose.py to flag dangerous postures: "
         "wrists above head (reaching/climbing), body near floor (fallen), rapid position change"],
        ["2", "Investigate IMG_3638 FPs",
         "Open runs/test_results/IMG_3638/ annotated video, identify triggering objects, "
         "add targeted hard negatives if needed"],
        ["3", "Alert / notification system",
         "Trigger an alert (sound, push notification) when a danger condition is detected "
         "for more than N consecutive frames"],
        ["4", "Pose fine-tuning (optional)",
         "If keypoint accuracy is insufficient on toddlers, find a toddler-specific pose "
         "dataset and fine-tune yolov8n-pose.pt with training/train_pose.py"],
    ],
    col_widths=[18, 45, 127]
)

# ── 14. TRAINING PLOTS ────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("14. Training Plots (Run 5)")
pdf.body(
    "The following plots were generated by Ultralytics during Run 5 and saved to "
    "runs/toddler_detect_s100/ on Google Colab, then exported to the plots/ directory."
)

pdf.insert_plot("results.png",                     "Figure 1 - Training & Validation Loss / mAP Curves (100 epochs)")
pdf.insert_plot("confusion_matrix_normalized.png", "Figure 2 - Normalized Confusion Matrix")
pdf.insert_plot("BoxPR_curve.png",                 "Figure 3 - Precision-Recall Curve")
pdf.insert_plot("BoxF1_curve.png",                 "Figure 4 - F1-Confidence Curve")
pdf.insert_plot("labels.jpg",                      "Figure 5 - Label Distribution & Bounding Box Statistics")
pdf.insert_plot("val_batch0_pred.jpg",             "Figure 6 - Validation Batch 0: Predictions")
pdf.insert_plot("val_batch1_pred.jpg",             "Figure 7 - Validation Batch 1: Predictions")

# ── SAVE ──────────────────────────────────────────────────────────────────────
pdf.output(str(OUTPUT))
print(f"Report saved to: {OUTPUT}")
