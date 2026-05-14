"""
YOLOv8 Toddler Detection - PDF Report Generator
Run: pip install fpdf2
Then: python inference/generate_report.py
Plots (optional): unzip plots.zip from Google Drive into <project_root>/plots/
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

    def table(self, headers, rows, col_widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(50, 50, 50)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for ri, row in enumerate(rows):
            self.set_fill_color(245, 245, 245 if ri % 2 == 0 else 255)
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
    "toddler detection dataset, covering data preparation, iterative training improvements, "
    "hard negative mining to eliminate adult false positives, and final evaluation.",
    align="C"
)

# ── 1. PROJECT OVERVIEW ───────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("1. Project Overview")
pdf.body(
    "The objective is to detect toddlers (young children) in video footage using a single-class "
    "YOLOv8s object detection model. The model outputs bounding boxes around every toddler visible "
    "in each frame. A major challenge was that the initial model falsely detected adults as toddlers "
    "due to the absence of adult examples during training. This was resolved through targeted hard "
    "negative mining."
)

pdf.section_title("2. Tools & Technologies")
pdf.kv("Model architecture",  "YOLOv8s - 11.1M parameters, 28.6 GFLOPs")
pdf.kv("Framework",           "Ultralytics 8.4.48 / PyTorch 2.10 / CUDA 12.8")
pdf.kv("Training hardware",   "Google Colab - Tesla T4 GPU (14.9 GB VRAM)")
pdf.kv("Detection class",     "toddler (single class, id=0)")
pdf.kv("Dataset format",      "YOLO format (.jpg images + .txt labels: class cx cy w h)")
pdf.kv("Hard neg. source 1",  "Roboflow - Children vs Adults (workspace: a-4euhx, version 4)")
pdf.kv("Hard neg. source 2",  "Custom - adult-only video frames extracted locally")
pdf.kv("Inference",           "inference/detect.py - supports video file, image, or webcam (conf=0.4)")

# ── 2. DATASET ────────────────────────────────────────────────────────────────
pdf.section_title("3. Dataset Preparation")
pdf.body("The dataset was prepared in three stages before the final training run:")
pdf.body(
    "Stage 1 - Re-split (70 / 15 / 15)\n"
    "All images and labels were pooled together and randomly re-split using seed=42 into "
    "70% train, 15% validation, 15% test. The original split had a large val/test gap (0.948 vs 0.700 "
    "mAP50) because validation images came from the same video clips as training images. "
    "After re-splitting the gap dropped to ~2%."
)
pdf.body(
    "Stage 2 - Roboflow hard negatives\n"
    "Adult-only images from the 'Children vs Adults' Roboflow dataset were downloaded and filtered: "
    "only images with no child annotations were kept. These were copied to the train set with empty "
    "label files (background class), teaching the model to suppress adult detections."
)
pdf.body(
    "Stage 3 - Custom video hard negatives\n"
    "Frames were extracted (1 per 2 seconds) from a real deployment video containing only adults. "
    "After manual review, all frames were added to the train set with empty labels. This targeted "
    "the specific visual environment of the deployment scenario."
)

pdf.table(
    headers=["Split", "Total Images", "Background Images", "Labeled Images"],
    rows=[
        ["train", "2018", "723", "1295"],
        ["val",   "289",  "8",   "281"],
        ["test",  "290",  "7",   "283"],
        ["TOTAL", "2597", "738", "1859"],
    ],
    col_widths=[35, 45, 55, 55]
)

# ── 3. TRAINING CONFIGURATION ─────────────────────────────────────────────────
pdf.section_title("4. Training Configuration")
pdf.table(
    headers=["Parameter", "Value"],
    rows=[
        ["Base weights",        "yolov8s.pt (COCO pretrained)"],
        ["Epochs",              "100"],
        ["Image size",          "640 x 640 px"],
        ["Batch size",          "32"],
        ["Optimizer",           "AdamW - lr=0.002, momentum=0.9, weight_decay=0.0005"],
        ["Early stopping",      "patience=15 epochs"],
        ["LR schedule",         "Cosine decay (lrf=0.01)"],
        ["Warmup",              "3 epochs"],
        ["Augmentations",       "Mosaic, flip LR (0.5), HSV, Blur, MedianBlur, CLAHE, Erasing"],
        ["AMP",                 "Enabled"],
        ["Close mosaic",        "Last 10 epochs"],
        ["Training time",       "~1.1 hours per run (Tesla T4)"],
    ],
    col_widths=[65, 125]
)

# ── 4. TRAINING ITERATIONS ────────────────────────────────────────────────────
pdf.section_title("5. Training Iterations & Results")
pdf.table(
    headers=["Run", "Description", "Val mAP50", "Test mAP50", "Test mAP50-95"],
    rows=[
        ["1", "YOLOv8n, original naive split",         "0.948", "0.700", "0.310"],
        ["2", "YOLOv8s, proper 70/15/15 re-split",     "0.956", "0.936", "0.537"],
        ["3", "Run 2 + Roboflow adult hard negatives",  "0.971", "0.951", "0.565"],
        ["4", "Run 3 + custom video hard negatives",    "0.961", "0.951", "0.567"],
    ],
    col_widths=[10, 90, 27, 28, 35]
)
pdf.body(
    "Key takeaways:\n"
    "  Run 1 -> 2: The val/test gap dropped from 24.8% to 2.0% by fixing the data split bias.\n"
    "  Run 2 -> 3: Roboflow hard negatives improved mAP50 across both val and test sets.\n"
    "  Run 3 -> 4: Custom environment-specific hard negatives eliminated adult false positives "
    "while keeping test accuracy stable at 0.951."
)

# ── 5. FINAL RESULTS ──────────────────────────────────────────────────────────
pdf.section_title("6. Final Model Results (Run 4 - best.pt)")
pdf.table(
    headers=["Metric", "Value"],
    rows=[
        ["Precision",           "0.943"],
        ["Recall",              "0.921"],
        ["Val mAP50",           "0.961"],
        ["Val mAP50-95",        "0.575"],
        ["Test mAP50",          "0.951"],
        ["Test mAP50-95",       "0.567"],
        ["Inference speed",     "~6 ms / image (Tesla T4)"],
        ["Model file size",     "22.5 MB"],
        ["Parameters",          "11,125,971"],
    ],
    col_widths=[90, 100]
)

# ── 6. PROBLEMS & SOLUTIONS ───────────────────────────────────────────────────
pdf.section_title("7. Problems Encountered & Solutions")
pdf.table(
    headers=["Problem", "Root Cause", "Solution"],
    rows=[
        ["Local train crashed\n(exit code 255)",
         "Windows PyTorch multiprocessing crash",
         "workers=0, amp=False,\nfreeze_support()"],
        ["Val/test gap 24.8%\n(0.948 vs 0.700)",
         "Val images from same video clips as train",
         "Merge all data, re-split 70/15/15 uniformly"],
        ["Adults detected as toddlers\n(conf up to 0.84)",
         "No adult examples in training data",
         "Hard negative mining with empty label files"],
        ["Toddler missed at conf=0.85",
         "Adult & toddler confidence scores overlap",
         "Reverted to conf=0.4, retrained with\ncustom hard negatives"],
    ],
    col_widths=[50, 70, 70]
)

# ── 7. PROJECT FILES ──────────────────────────────────────────────────────────
pdf.section_title("8. Project Files")
pdf.table(
    headers=["File", "Description"],
    rows=[
        ["config.py",                              "Central path configuration (project root)"],
        ["notebooks/train_colab.ipynb",            "Google Colab training notebook (10 cells)"],
        ["training/train_detect.py",               "Local detection training script (CPU / Windows-safe)"],
        ["training/train_pose.py",                 "Local pose training script (SyRIP dataset)"],
        ["inference/detect.py",                    "Inference script - video / image / webcam"],
        ["inference/generate_report.py",           "This report generator"],
        ["data/prepare_dataset.py",                "One-time 80/20 train/val split utility"],
        ["data/extract_hard_negatives.py",         "Extracts frames from video as hard negatives"],
        ["data/download_syrip.py",                 "Downloads the SyRIP infant pose dataset"],
        ["data/convert_syrip_to_yolo.py",          "Converts SyRIP COCO keypoints to YOLO pose format"],
        ["models/toddler_detect_s100_best.pt",     "Final detection model weights - 22.5 MB"],
        ["dataset/detect_child/data.yaml",         "YOLO detection dataset configuration"],
        ["dataset/pose_child/data.yaml",           "YOLO pose dataset configuration"],
        ["adult_negatives/",                       "Custom adult hard negative frames"],
    ],
    col_widths=[75, 115]
)

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("9. Training Plots")
pdf.body(
    "The following plots were generated by Ultralytics during training and saved to "
    "/content/runs/toddler_detect_s100/ on Google Colab.\n"
    "To include them: unzip plots.zip from Google Drive into <project_root>/plots/"
)

pdf.insert_plot("results.png",                     "Figure 1 - Training & Validation Curves (loss, mAP50, mAP50-95)")
pdf.insert_plot("confusion_matrix_normalized.png", "Figure 2 - Confusion Matrix (Normalized)")
pdf.insert_plot("PR_curve.png",                    "Figure 3 - Precision-Recall Curve")
pdf.insert_plot("F1_curve.png",                    "Figure 4 - F1-Confidence Curve")
pdf.insert_plot("labels.jpg",                      "Figure 5 - Label Distribution & Bounding Box Statistics")

# ── SAVE ──────────────────────────────────────────────────────────────────────
pdf.output(str(OUTPUT))
print(f"Report saved to: {OUTPUT}")
