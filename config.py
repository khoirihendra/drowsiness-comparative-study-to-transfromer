"""
Centralized Configuration for Drowsiness Detection Research (UTA-RLDD).

This module contains all path configurations, feature extraction parameters,
model architectures, training hyperparameters, and reproducibility settings.
"""

import os
from pathlib import Path

# ==========================================
# 1. BASE DIRECTORIES & PATHS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
FEATURES_DIR = OUTPUT_DIR / "extracted_features"

# Ensure all output directories exist
for directory in [OUTPUT_DIR, CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR, FEATURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default dataset path (override via CLI or environment variable)
DEFAULT_DATASET_PATH = os.environ.get(
    "UTA_RLDD_DATASET_PATH",
    "/kaggle/input/datasets/rishab260/uta-reallife-drowsiness-dataset"
)

# MediaPipe face landmarker model path
DEFAULT_LANDMARKER_MODEL_PATH = os.environ.get(
    "MEDIAPIPE_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "face_landmarker.task")
)

# ==========================================
# 2. REPRODUCIBILITY
# ==========================================
SEED = 42

# ==========================================
# 3. DATASET & FOLD SPECIFICATIONS
# ==========================================
# Label mappings according to UTA-RLDD dataset naming conventions:
# 0.mp4 / 0.mov / 0_glasses.mp4  -> 0 (Alert)
# 5.mp4 / 5.mov / 5_glasses.mp4  -> 1 (Low Vigilant)
# 10.mp4 / 10.mov / 10_glasses.mp4 -> 2 (Drowsy)
LABEL_MAP = {
    "0": 0,    # Alert
    "5": 1,    # Low Vigilant
    "10": 2,   # Drowsy
}

LABEL_NAMES = ["Alert", "Low Vigilant", "Drowsy"]
NUM_CLASSES = 3

# UTA-RLDD Official 5-Fold Cross-Validation Subject Mapping (60 subjects total)
# Fold 1: Subjects 1 - 12
# Fold 2: Subjects 13 - 24
# Fold 3: Subjects 25 - 36
# Fold 4: Subjects 37 - 48
# Fold 5: Subjects 49 - 60
FOLD_SUBJECT_MAPPING = {
    1: list(range(1, 13)),
    2: list(range(13, 25)),
    3: list(range(25, 37)),
    4: list(range(37, 49)),
    5: list(range(49, 61)),
}
TOTAL_FOLDS = 5

# ==========================================
# 4. FEATURE EXTRACTION PARAMETERS
# ==========================================
FRAME_SKIP = 5                # Sample 1 frame every N frames
MAX_FRAMES_PER_VIDEO = 5000   # Max sampled frames per video (safety cutoff)
FRAME_RESIZE = (640, 480)     # Width, Height for face landmarking

# Feature definitions (5 features extracted per frame):
# Index 0: Eye Aspect Ratio (EAR)
# Index 1: Mouth Aspect Ratio (MAR)
# Index 2: Pitch (Head Pose)
# Index 3: Yaw (Head Pose)
# Index 4: Roll (Head Pose)
FEATURE_NAMES = ["EAR", "MAR", "Pitch", "Yaw", "Roll"]
TOTAL_RAW_FEATURES = 5

# Placeholder emitted when MediaPipe cannot detect a face in a sampled frame.
# Keep this definition centralized so extraction and integrity checks cannot drift.
MISSING_FEATURE_VECTOR = (0.3, 0.0, 0.0, 0.0, 0.0)
MAX_ALLOWED_PADDING_RATE = 0.50

# Feature subset slicing configurations:
FEATURE_SUBSETS = {
    "ear": {
        "indices": [0],
        "name": "EAR Only",
        "num_features": 1,
    },
    "ear_mar": {
        "indices": [0, 1],
        "name": "EAR + MAR",
        "num_features": 2,
    },
    "all": {
        "indices": [0, 1, 2, 3, 4],
        "name": "EAR + MAR + HeadPose (5 Features)",
        "num_features": 5,
    }
}

# ==========================================
# 5. TEMPORAL SEQUENCE / SLIDING WINDOW
# ==========================================
SEQ_LENGTH = 30               # Sequence length (timesteps) per sample
STEP_SIZE = 1                 # Stride for sliding window generation (1 for dense overlap)

# ==========================================
# 6. MODEL ARCHITECTURES & HYPERPARAMETERS
# ==========================================
SUPPORTED_MODELS = [
    "bilstm",
    "lstm",
    "bigru",
    "cnn1d",
    "transformer",
    "xgboost"
]

# Deep Learning Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
L2_REGULARIZATION = 0.001
DROPOUT_RATE = 0.4
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# Transformer-specific Hyperparameters
TRANSFORMER_HEAD_SIZE = 64
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_NUM_BLOCKS = 2
TRANSFORMER_DROPOUT = 0.3

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 15,
}
