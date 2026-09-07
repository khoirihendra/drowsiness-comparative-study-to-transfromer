# 😴 UTA-RLDD Drowsiness Detection: Reproducible End-to-End Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-brightgreen.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A publication-ready, fully reproducible research codebase for driver drowsiness detection using the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**.

---

## 📌 Key Highlights

- **Strict Zero Data Leakage**: Temporal sliding windows are partitioned at the individual video/subject level. 5-Fold Cross-Validation strictly isolates subjects across train, validation, and test sets.
- **Official UTA-RLDD 5-Fold Evaluation Protocol**: Evaluates all 5 folds by leaving one fold out for testing and averaging the resulting metrics ($mean \pm std$), making all results directly comparable with standard literature.
- **Spatial Micro-Expression & 3D Pose Extraction**: Extracted using Google MediaPipe Face Landmarker / FaceMesh:
  1. **EAR (Eye Aspect Ratio)**: Eye openness and blink duration dynamics.
  2. **MAR (Mouth Aspect Ratio)**: Yawning frequency and duration.
  3. **Pitch (Head Pose)**: Nodding movements associated with microsleep.
  4. **Yaw (Head Pose)**: Inattentive head turning.
  5. **Roll (Head Pose)**: Lateral head dropping towards shoulders.
- **Multi-Model Benchmark Suite**:
  - `BiLSTM` (Bidirectional Long Short-Term Memory)
  - `LSTM` (Standard Long Short-Term Memory)
  - `BiGRU` (Bidirectional Gated Recurrent Unit)
  - `1D-CNN` (1D Convolutional Neural Network)
  - `Transformer` (Multi-Head Self-Attention for Time-Series)
  - `XGBoost` (Gradient Boosted Decision Trees baseline)
- **Feature Ablation Studies**: Supported presets for `EAR Only` (1 feature), `EAR + MAR` (2 features), and `5 Features (EAR + MAR + Pitch + Yaw + Roll)`.

---

## 📂 Repository Structure

```
refactor/
├── README.md                      # Complete project documentation & guide
├── requirements.txt               # Dependencies with version constraints
├── config.py                      # Centralized configuration (hyperparameters, paths, seed)
├── extract_video_features.py      # Reusable frame-level MediaPipe cache
├── build_windows.py               # Cheap temporal configuration builder
├── extract_features.py            # Legacy one-pass extraction + windowing
├── train.py                       # 5-Fold Cross-Validation training script
├── evaluate.py                    # Evaluation & comparison tables/figures generator
├── run_all_experiments.py         # Automated ablation experiment runner
├── drowsiness_experiment.ipynb    # Interactive notebook (Kaggle / Colab ready)
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Global seed, metrics computation, plotting helpers
│   ├── dataset.py                 # Subject-aware dataset loader & sliding-window generator
│   ├── feature_extractor.py       # MediaPipe Landmarker, EAR, MAR, solvePnP Head Pose
│   └── models/
│       ├── __init__.py            # Model registry
│       ├── bilstm.py              # Bidirectional LSTM architecture
│       ├── lstm.py                # Standard LSTM architecture
│       ├── bigru.py               # Bidirectional GRU architecture
│       ├── cnn1d.py               # 1D-CNN architecture
│       ├── transformer.py         # Time-Series Transformer architecture
│       └── xgboost_model.py       # XGBoost classifier with temporal flattening
└── output/                        # (Auto-generated during execution)
    ├── extracted_features/        # Preprocessed .npz feature archives
    ├── checkpoints/               # Saved best model weights (.keras / .json)
    ├── metrics/                   # Per-fold and 5-fold CV summary JSONs
    └── figures/                   # Learning curves, confusion matrices, comparison charts
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/drowsiness-detection-uta-rldd.git
cd drowsiness-detection-uta-rldd/refactor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation (UTA-RLDD)

Download the dataset from Kaggle or official sources:
- **Kaggle Dataset**: [UTA Real-Life Drowsiness Dataset](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)
- **Missing Fold 5 supplement**: [UTA-RLDD Fold 5](https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-fold5)

> **Important:** the first Kaggle upload contains folds 1-4 only. Add the Fold 5
> supplement as a second Kaggle input for the complete 60-subject benchmark.

The dataset contains videos for 60 subjects across 3 drowsiness states:
- `0.mp4` / `0.mov`: **Alert** (Label 0)
- `5.mp4` / `5.mov`: **Low Vigilant** (Label 1)
- `10.mp4` / `10.mov`: **Drowsy** (Label 2)

**Subject to Fold Partitioning (Official Benchmark Protocol)**:
- **Fold 1**: Subjects 01 - 12
- **Fold 2**: Subjects 13 - 24
- **Fold 3**: Subjects 25 - 36
- **Fold 4**: Subjects 37 - 48
- **Fold 5**: Subjects 49 - 60

---

## 🚀 Usage Guide

### Step 1: Extract the reusable frame-level cache

Run MediaPipe once for every decoded frame. The command is resumable: completed
per-video cache files are reused if the same command is interrupted and rerun.

```bash
python extract_video_features.py \
    --dataset_path /path/to/uta-rldd-folds-1-to-4 /path/to/uta-rldd-fold-5 \
    --cache_dir output/video_cache \
    --model_path models/face_landmarker.task \
    --num_workers 4
```

Use `--require_explicit_fold` when every source path contains a `Fold1` through
`Fold5` directory. Without it, inferred fold assignments are explicitly reported
in `manifest.json`. Do not use `--max_frames` for the final research dataset; it is
provided only for quick pipeline checks. By default the extractor also requires
the canonical inventory of 180 videos, 60 subjects, three labels per subject, and
12 subjects per fold. `--allow_incomplete_dataset` relaxes this only for deliberate
partial-data or debug runs.

### Step 2: Build temporal configurations without MediaPipe

Each command below reads the same frame cache. `step_size` is measured in sampled
timesteps, after `frame_skip` has been applied.

```bash
python build_windows.py \
    --cache_dir output/video_cache \
    --output_path output/extracted_features/fs5_seq30_step1.npz \
    --frame_skip 5 \
    --sequence_length 30 \
    --step_size 1
```

Another configuration only requires another inexpensive builder command:

```bash
python build_windows.py \
    --cache_dir output/video_cache \
    --output_path output/extracted_features/fs3_seq60_step10.npz \
    --frame_skip 3 \
    --sequence_length 60 \
    --step_size 10
```

The resulting NPZ remains compatible with `train.py` and additionally stores
video IDs, detection masks, original frame ranges, timestamps, and build metadata
for leakage and overlap audits.

### Legacy alternative: one-pass extraction

The older command below still works, but bakes `frame_skip`, sequence length, and
step size into its output and therefore cannot reuse MediaPipe results across a
temporal grid:

```bash
python extract_features.py \
    --dataset_path /path/to/uta-rldd-folds-1-to-4 /path/to/uta-rldd-fold-5 \
    --output_path output/extracted_features/uta_rldd_features_seq30.npz \
    --frame_skip 5 \
    --seq_length 30
```

Extraction now performs an integrity check before saving. It aborts if MediaPipe
produces constant features or if more than 50% of sampled frames use missing-face
padding. Change the latter threshold explicitly with `--max_padding_rate` only after
inspecting the affected videos.

### Step 3: Training & 5-Fold Cross-Validation
Train any model with a specific feature subset across all 5 folds:
```bash
# Example 1: Train BiLSTM with all 5 features (Full 5-Fold CV)
python train.py --model bilstm --features all --data_path output/extracted_features/uta_rldd_features_seq30.npz

# Example 2: Train Transformer with EAR + MAR (Fold 1 only)
python train.py --model transformer --features ear_mar --fold 1 --data_path output/extracted_features/uta_rldd_features_seq30.npz

# Example 3: Train XGBoost baseline
python train.py --model xgboost --features all --data_path output/extracted_features/uta_rldd_features_seq30.npz
```

### Step 4: Benchmark Evaluation & Summary Generation
Compile all metrics into a Markdown/LaTeX table and generate comparison charts:
```bash
python evaluate.py
```

### Step 5: Run All Ablation Experiments Automatically
Run the complete grid across all models (`bilstm`, `lstm`, `bigru`, `cnn1d`, `transformer`, `xgboost`) and feature sets (`ear`, `ear_mar`, `all`):
```bash
python run_all_experiments.py --data_path output/extracted_features/uta_rldd_features_seq30.npz
```

---

## 📈 Paper Benchmark Template

Results after running 5-Fold Cross-Validation ($Mean \pm Std$):

| Model | Feature Set | Accuracy (%) | Macro F1 (%) | Precision (%) | Recall (%) |
|:---|:---|:---:|:---:|:---:|:---:|
| **BiLSTM** | EAR Only | -- | -- | -- | -- |
| **BiLSTM** | EAR + MAR | -- | -- | -- | -- |
| **BiLSTM** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |
| **Transformer** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |
| **1D-CNN** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |
| **BiGRU** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |
| **LSTM** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |
| **XGBoost** | 5 Features (EAR+MAR+Pose) | -- | -- | -- | -- |

---

## 🔒 Zero Data Leakage Guarantee

1. **Temporal Slicing**: Windows of length $T=30$ are generated *per video*, avoiding temporal boundary overlap across videos.
2. **Subject Partitioning**: Data is split on the subject level before feeding to model pipelines. No video or subject from the test fold appears in the training or validation sets.
3. **Reproducibility**: `set_seed(42)` ensures deterministic splits, model initializations, and shuffling.

---

## 📜 Citation

If you use this codebase or the UTA-RLDD dataset in your research, please cite:

```bibtex
@inproceedings{Ghoddoosian2019Realistic,
  title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
  author={Ghoddoosian, Reza and Galib, Mohammad and Athitsos, Vassilis},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2019}
}
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
