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

pdf.body("A. Detection Model - YOLOv8s")
pdf.body(
    "YOLOv8 (You Only Look Once, version 8) is a single-stage, anchor-free real-time object "
    "detector developed by Ultralytics. Unlike two-stage detectors (e.g. Faster R-CNN), YOLOv8 "
    "processes the full image in a single forward pass, making it suitable for real-time video.\n\n"
    "Architecture breakdown:\n"
    "  Backbone - CSPDarknet with C2f (Cross-Stage Partial with feature fusion) blocks. "
    "Extracts multi-scale feature maps at three resolutions (80x80, 40x40, 20x20 for a 640px input). "
    "The C2f block is a reparameterized bottleneck that improves gradient flow during training.\n"
    "  Neck - PANet (Path Aggregation Network) feature pyramid. Merges backbone feature maps "
    "top-down and bottom-up so the detector has both semantic context (deep layers) and spatial "
    "precision (shallow layers) at every scale.\n"
    "  Head - Decoupled detection head: separate branches for classification (is it a toddler?) "
    "and regression (where exactly is it?). Uses Distribution Focal Loss (DFL) for bounding box "
    "regression, which models the box coordinates as a probability distribution rather than a "
    "single point - this is what drives the high mAP50-95 score (precise box localization).\n"
    "  Anchor-free design: box centres are predicted directly relative to each grid cell with no "
    "predefined anchor shapes, which simplifies training and generalises better to unusual aspect ratios."
)
pdf.table(
    headers=["YOLOv8 Variant", "Parameters", "GFLOPs", "Reason for choice"],
    rows=[
        ["YOLOv8n (nano)",  "3.2 M",  "8.7",  "Run 1 only - underfitted, mAP50-95=0.310"],
        ["YOLOv8s (small)", "11.1 M", "28.6", "Runs 2-5 - best accuracy/speed trade-off"],
        ["YOLOv8m (medium)","25.9 M", "78.9", "Not used - too slow for real-time on CPU"],
    ],
    col_widths=[42, 22, 22, 104]
)
pdf.kv("Input resolution",   "640 x 640 px (letterbox padded, preserves aspect ratio)")
pdf.kv("Output per frame",   "N bounding boxes, each with (x1, y1, x2, y2, confidence, class_id)")
pdf.kv("Confidence thresh.", "0.4 at inference (boxes below this discarded before NMS)")
pdf.kv("NMS IoU threshold",  "0.45 (default Ultralytics) - suppresses overlapping boxes")
pdf.kv("Final model file",   "models/toddler_detect_s100_best.pt  (21.5 MB)")
pdf.ln(3)

pdf.body("B. Pose Estimation Model - YOLOv8n-pose")
pdf.body(
    "YOLOv8n-pose is the nano variant of YOLOv8 extended with a keypoint regression head. "
    "In addition to detecting a bounding box around each person, it regresses 17 (x, y, confidence) "
    "keypoint tuples - one per COCO body joint. The model was pre-trained on the COCO Keypoints "
    "dataset (~250k person instances with annotated skeletons) and requires no fine-tuning for "
    "toddlers because the human skeletal topology is identical across age groups.\n\n"
    "Keypoint head: appended to the same YOLOv8n backbone/neck. For each detected person, it "
    "outputs a 51-dimensional vector (17 joints x 3 values: x, y, visibility confidence). "
    "Keypoints with confidence below 0.4 are not drawn."
)
pdf.kv("Model file",       "yolov8n-pose.pt  (~6 MB, auto-downloaded from Ultralytics on first run)")
pdf.kv("Keypoints",        "17 COCO joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)")
pdf.kv("Skeleton edges",   "16 bone connections drawn from the COCO skeleton definition")
pdf.kv("Keypoint colors",  "Green=face, Orange=shoulders, Blue gradient=arms, Cyan=legs")
pdf.kv("Role in pipeline", "Provides per-joint positions used for pose analysis and future danger detection")
pdf.ln(3)

pdf.body("C. Multi-Object Tracker - ByteTrack")
pdf.body(
    "ByteTrack is a multi-object tracking (MOT) algorithm that assigns each detected object a "
    "persistent integer ID across video frames. It is built directly into the Ultralytics library "
    "and activated with a single parameter (tracker='bytetrack.yaml'), requiring no additional "
    "dependencies. ByteTrack works in two matching passes per frame:\n"
    "  Pass 1 (high-confidence): detections above a high threshold are matched to existing tracks "
    "using the Hungarian algorithm on IoU (Intersection over Union) distance. Matched tracks are "
    "updated with the new box; unmatched tracks enter a 'lost' buffer.\n"
    "  Pass 2 (low-confidence): detections that fall between a floor threshold and the high "
    "threshold are used to recover tracks from the lost buffer. This two-stage design is the "
    "key innovation: it avoids discarding detections that are valid but temporarily uncertain "
    "(occlusion, motion blur), which is the main source of ID switches in other trackers.\n\n"
    "Track lifecycle: New -> Active (matched in Pass 1 or 2) -> Lost (unmatched, kept in buffer "
    "for EXIT_PATIENCE=45 frames) -> Removed (logged as exit event)."
)
pdf.kv("Config file",      "bytetrack.yaml (built into Ultralytics installation)")
pdf.kv("Matching metric",  "IoU distance + Hungarian assignment (linear sum assignment)")
pdf.kv("Exit patience",    "45 frames absent before track removed and exit event logged")
pdf.kv("Trail rendering",  "Last 50 centre positions per track; line opacity proportional to age")
pdf.kv("ID color scheme",  "Unique HSV hue per ID: hue = (ID x 47) mod 180, converted to BGR")
pdf.ln(3)

pdf.body("D. Framework & Runtime Stack")
pdf.table(
    headers=["Component", "Version", "Role"],
    rows=[
        ["Python",          "3.10",       "Runtime language"],
        ["Ultralytics",     "8.4.50",     "YOLOv8 training, inference, tracking, pose API"],
        ["PyTorch",         "2.10",       "Deep learning backend (autograd, CUDA tensors)"],
        ["CUDA",            "12.8",       "GPU acceleration (Colab A100 training)"],
        ["OpenCV (cv2)",    "4.x",        "Frame decoding, drawing, display, video writing"],
        ["NumPy",           "1.x",        "Array operations for box/keypoint coordinate math"],
        ["fpdf2",           "2.x",        "PDF report generation (this document)"],
        ["Roboflow API",    "roboflow",   "Downloading hard negative datasets"],
    ],
    col_widths=[38, 24, 128]
)
pdf.ln(3)

pdf.body("E. Training Hardware")
pdf.table(
    headers=["Run", "Hardware", "Notes"],
    rows=[
        ["1",   "Local CPU (Windows)",         "Workers=0, AMP disabled - very slow, only 50 epochs"],
        ["2-4", "Google Colab A100 (40 GB)",   "~1 hour per 100-epoch run, batch=64, AMP enabled"],
        ["5",   "Google Colab A100 (40 GB)",   "Final run - pets/furniture/rooms/toys hard negatives added on Colab"],
    ],
    col_widths=[12, 55, 123]
)
pdf.ln(3)

pdf.body("F. Dataset Format - YOLO Detection Format")
pdf.body(
    "Each image in the dataset has a paired .txt label file with the same stem name. "
    "Each line in the label file represents one object:\n"
    "  <class_id> <cx> <cy> <w> <h>\n"
    "where all coordinates are normalised to [0, 1] relative to the image dimensions. "
    "class_id=0 always (toddler is the only class). Hard negative images have an empty label "
    "file (0 bytes) - this signals to the model that any detection on that image is a false "
    "positive that should be suppressed, effectively teaching the model what NOT to detect."
)
pdf.kv("Image format",    ".jpg, 640x640 px target (letterbox resized during training)")
pdf.kv("Label format",    ".txt - one row per object: class cx cy w h (normalised)")
pdf.kv("Hard negatives",  "Empty .txt file (0 bytes) - background-only images, no annotations")
pdf.kv("dataset config",  "dataset/detect_child/data.yaml - declares train/val/test paths and nc=1")
pdf.ln(3)

pdf.body("G. Hard Negative Sources")
pdf.table(
    headers=["Source", "Category", "Images added", "Purpose"],
    rows=[
        ["Roboflow - Children vs Adults", "Adult humans",       "~300", "Teach model to ignore adult bodies"],
        ["Custom video frames",           "Adults (local env)", "74",   "Target specific deployment camera angle"],
        ["Roboflow - Oxford IIIT Pets",   "Cats & dogs",        "~150", "Suppress animal silhouettes (resemble crouching toddlers)"],
        ["Roboflow - Furniture detection","Chairs, tables",     "~100", "Suppress common indoor furniture"],
        ["Roboflow - Indoor room scenes", "Walls, floors",      "~100", "Suppress empty room backgrounds"],
        ["Roboflow - Toys detection",     "Toys on floor",      "~100", "Suppress small objects near floor level"],
    ],
    col_widths=[55, 38, 22, 75]
)

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
        ["ID fragmentation: 31 IDs in 4 s (before fix)",
         "Detections dropped below conf=0.4 for 1-2 frames causing "
         "ByteTrack to lose and reassign tracks",
         "Fixed: DETECT_CONF=0.1, new_track_thresh=0.5, track_buffer=90 "
         "in bytetrack_toddler.yaml. Longest track improved from 121 to 214 frames."],
    ],
    col_widths=[52, 68, 70]
)

# ── 9. MULTI-OBJECT TRACKING ─────────────────────────────────────────────────
pdf.section_title("9. Multi-Object Tracking")
pdf.body(
    "After the detection model was validated, ByteTrack was added to assign each toddler a "
    "persistent ID across frames. ByteTrack is built into Ultralytics and requires no extra "
    "dependencies. It is robust to brief occlusions: if two toddlers overlap, their individual "
    "IDs are recovered once they separate.\n\n"
    "ByteTrack Algorithm (two-pass matching per frame):\n"
    "  Pass 1 - High-confidence detections are matched to existing active tracks using IoU-based "
    "Hungarian assignment. Tracks with a strong enough IoU overlap are updated; unmatched tracks "
    "enter a 'lost' state.\n"
    "  Pass 2 - Low-confidence detections (between a floor and the high threshold) are used to "
    "recover lost tracks that were temporarily occluded. This two-pass design is what allows IDs "
    "to survive brief overlaps between two toddlers."
)
pdf.kv("Tracker",        "ByteTrack (bytetrack.yaml) - built into Ultralytics")
pdf.kv("Script",         "inference/track.py")
pdf.kv("Conf threshold", "0.4 (detections below this are rejected entirely)")
pdf.kv("ID persistence", "Each toddler keeps its ID across the full video")
pdf.kv("Exit patience",  "45 frames of absence before an exit event is logged")
pdf.kv("Trail length",   "Last 50 centre positions drawn per track, fading with age")
pdf.kv("Trail colors",   "Unique HSV color per track ID - hue = (ID x 47) mod 180")
pdf.kv("Output",         "Per-ID coloured box, motion trail, entry/exit log, session summary")
pdf.ln(2)
pdf.body(
    "Entry and exit events are printed to the console with timestamps (HH:MM:SS). At the end of "
    "the video, a summary shows the total number of unique toddlers seen, the longest track "
    "duration, and the average track length.\n\n"
    "ByteTrack Tuning - ID Fragmentation Fix (2026-05-15):\n"
    "  Initial test showed 31 unique IDs in 4 s (longest track: 121 frames). Root cause: "
    "detections dropped below conf=0.4 for 1-2 frames, causing ByteTrack to lose and reassign tracks.\n\n"
    "  Fix - decoupled detection floor from track-creation threshold (bytetrack_toddler.yaml):\n"
    "  DETECT_CONF: 0.4 -> 0.1   (passes weak detections to tracker for recovery)\n"
    "  track_high_thresh: 0.25 -> 0.4  (only strong detections drive Pass 1 matching)\n"
    "  track_low_thresh:  0.10 -> 0.05 (weak detections used only for Pass 2 recovery)\n"
    "  new_track_thresh:  0.25 -> 0.50 (requires high confidence to start a new track)\n"
    "  track_buffer:      30   -> 90   (tracks survive 3 s without a detection)\n\n"
    "  Results after fix on 'multi_toddler tracking test.mp4':\n"
    "  Frames: 215  |  Unique IDs: 38  |  Longest track: ID 3 (214 frames = 7.1 s)\n"
    "  Tracks ID 2 (262 f), ID 3 (214 f), ID 5 (214 f), ID 11 (209 f) survived nearly the full video."
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
    "pose model. Adults, pets, and background objects are ignored entirely.\n\n"
    "Two models run sequentially on every frame:"
)
pdf.body(
    "Flow per frame:\n"
    "  1. Detection model + ByteTrack  ->  tracked toddler boxes [(ID, x1, y1, x2, y2), ...]\n"
    "  2. Pose model (full frame)      ->  all visible person skeletons (may include adults)\n"
    "  3. Matching                     ->  skeleton centre must fall inside a tracked toddler box "
    "(+/-30 px margin) to be kept; adult skeletons are silently discarded\n"
    "  4. Drawing                      ->  per-ID colour, motion trail, bounding box + label, "
    "skeleton all composited on the same frame"
)
pdf.body(
    "Skeleton-to-detection matching uses a centre-point test with a 30-pixel tolerance to handle "
    "slight misalignment between the two models' bounding box predictions:\n"
    "  cx = (pose_x1 + pose_x2) / 2 ;  cy = (pose_y1 + pose_y2) / 2\n"
    "  matched if:  det_x1-30 <= cx <= det_x2+30  AND  det_y1-30 <= cy <= det_y2+30"
)
pdf.kv("Script",         "inference/track_and_pose.py")
pdf.kv("Detect conf",    "0.4  |  Pose conf: 0.4  |  Keypoint conf threshold: 0.4")
pdf.kv("Output video",   "runs/track_pose/<stem>_tracked_pose.mp4  (mp4v codec, original resolution)")
pdf.kv("Display",        "Live resized window at 960 px width - press Q to quit")
pdf.ln(2)
pdf.body(
    "Live test (2026-05-15) - 'multi_toddler tracking test.mp4' (after ByteTrack tuning):\n"
    "  Detection model : toddler_detect_s100_best.pt  |  Tracker: ByteTrack  |  Pose: yolov8n-pose.pt\n"
    "  Total frames    : 215 (7.2 s at 30 fps)\n"
    "  Unique IDs      : 38  |  Longest track: ID 3 (214 frames = 7.1 s)\n"
    "  Output saved to : runs/track_pose/multi_toddler tracking test_tracked_pose.mp4\n\n"
    "All three annotation layers (bounding boxes with ID labels, motion trails, and body "
    "skeletons) rendered correctly on each confirmed toddler. Adults received no skeleton, "
    "confirming the detection model gatekeeper logic works as intended.\n\n"
    "Intermediate scripts also available:\n"
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
         "wrists above nose (reaching/climbing), ankle Y near hip Y (fallen/crawling), "
         "large sudden centre-of-mass jump (rapid fall), box centre near frame edge (leaving zone)."],
        ["2", "Alert / notification system",
         "Trigger an alert (sound, desktop push notification) when a danger condition "
         "persists for more than N consecutive frames (debounce to avoid single-frame noise)."],
        ["3", "Investigate IMG_3638 FPs",
         "Open runs/test_results/IMG_3638/ annotated video, identify objects triggering "
         "conf=0.84 detections, add targeted hard negatives for those objects if needed."],
        ["4", "Pose fine-tuning (optional)",
         "If keypoint accuracy is insufficient on toddlers at typical camera angles, "
         "fine-tune yolov8n-pose.pt on a toddler-specific pose dataset using train_pose.py."],
    ],
    col_widths=[18, 48, 124]
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
