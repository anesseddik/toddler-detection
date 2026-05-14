from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Datasets
DATASET_DETECT_DIR = ROOT / "dataset" / "detect_child"
DATASET_POSE_DIR   = ROOT / "dataset" / "pose_child"
SYRIP_RAW_DIR      = ROOT / "syrip_raw"

# Model artefacts
MODELS_DIR         = ROOT / "models"
FINAL_DETECT_MODEL = MODELS_DIR / "toddler_detect_s100_best.pt"

# Outputs
RUNS_DIR           = ROOT / "runs"
PLOTS_DIR          = ROOT / "plots"
REPORTS_DIR        = ROOT / "reports"

# Hard negatives
ADULT_NEG_DIR      = ROOT / "adult_negatives"
