"""
Training and 5-Fold Cross-Validation Script for Drowsiness Detection.

Implements the official UTA-RLDD 5-Fold Cross-Validation protocol
with STRICT ZERO SUBJECT LEAKAGE and full reproducibility.

Supported Models:
    - bilstm      : Bidirectional LSTM
    - lstm        : Standard LSTM
    - bigru       : Bidirectional GRU
    - cnn1d       : 1D Convolutional Neural Network
    - transformer : Time-Series Multi-Head Attention Transformer
    - xgboost     : XGBoost Classifier (with temporal sequence flattening)

Supported Feature Subsets:
    - ear         : 1 Feature (EAR only)
    - ear_mar     : 2 Features (EAR + MAR)
    - all         : 5 Features (EAR + MAR + Pitch + Yaw + Roll)

Usage:
    # Run full 5-fold CV for BiLSTM with all 5 features:
    python train.py --model bilstm --features all --data_path output/extracted_features/uta_rldd_features_seq30.npz

    # Run single fold (e.g. Fold 1 test):
    python train.py --model transformer --features ear_mar --fold 1 --data_path data.npz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FEATURE_SUBSETS,
    FIGURES_DIR,
    L2_REGULARIZATION,
    LABEL_NAMES,
    LEARNING_RATE,
    METRICS_DIR,
    MIN_LR,
    NUM_CLASSES,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
    SUPPORTED_MODELS,
    TOTAL_FOLDS,
    FEATURES_DIR
)
from src.utils import (
    set_seed,
    compute_metrics,
    plot_learning_curves,
    plot_confusion_matrix,
    save_json
)
from src.dataset import UTARLDDDataset
from src.models import MODEL_BUILDERS
from src.models.xgboost_model import train_xgboost_model, evaluate_xgboost_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train drowsiness detection models with UTA-RLDD 5-Fold Cross-Validation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        choices=SUPPORTED_MODELS,
        help=f"Model architecture to train: {SUPPORTED_MODELS}"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        choices=list(FEATURE_SUBSETS.keys()),
        help=f"Feature subset to use: {list(FEATURE_SUBSETS.keys())}"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(FEATURES_DIR / "uta_rldd_features_seq30.npz"),
        help="Path to pre-extracted .npz or .npy dataset file."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Test fold index (1-5) to train single fold, or 0 to run all 5 folds."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum training epochs (default: {EPOCHS})."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Initial learning rate (default: {LEARNING_RATE})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})."
    )
    return parser.parse_args()


def train_single_fold_nn(
    model_name: str,
    feature_subset: str,
    test_fold: int,
    dataset: UTARLDDDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int
) -> Dict:
    """Train and evaluate a Deep Learning model on one cross-validation fold."""
    set_seed(seed)

    # Prepare zero-leakage fold data
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, y_test_raw) = dataset.get_fold_split(
        test_fold=test_fold,
        feature_subset=feature_subset,
        one_hot=True
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    builder = MODEL_BUILDERS[model_name]
    model = builder(
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        l2_reg=L2_REGULARIZATION,
        dropout_rate=DROPOUT_RATE
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_{feature_subset}_fold{test_fold}_best.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        )
    ]

    print(f"\n--- Training {model_name.upper()} | Feature: {feature_subset} | Test Fold: {test_fold} ---")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print(f"Input Shape  : {input_shape}\n")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # Evaluate on held-out test fold
    y_pred_probs = model.predict(X_test, verbose=0)
    metrics = compute_metrics(y_test_raw, y_pred_probs, class_names=LABEL_NAMES)
    metrics["test_fold"] = test_fold
    metrics["model"] = model_name
    metrics["feature_subset"] = feature_subset
    metrics["checkpoint_path"] = str(checkpoint_path)

    # Save curve and confusion matrix figures
    curve_fig_path = FIGURES_DIR / f"{model_name}_{feature_subset}_fold{test_fold}_learning_curves.png"
    cm_fig_path = FIGURES_DIR / f"{model_name}_{feature_subset}_fold{test_fold}_confusion_matrix.png"

    plot_learning_curves(
        history.history,
        title=f"{model_name.upper()} ({feature_subset}) - Fold {test_fold}",
        save_path=curve_fig_path
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=LABEL_NAMES,
        title=f"{model_name.upper()} - Fold {test_fold} (Acc: {metrics['accuracy']*100:.2f}%)",
        save_path=cm_fig_path
    )

    return metrics


def train_single_fold_xgboost(
    feature_subset: str,
    test_fold: int,
    dataset: UTARLDDDataset,
    seed: int
) -> Dict:
    """Train and evaluate XGBoost baseline model on one fold."""
    set_seed(seed)

    (X_train, y_train, _), (X_val, y_val, _), (X_test, _, y_test_raw) = dataset.get_fold_split(
        test_fold=test_fold,
        feature_subset=feature_subset,
        one_hot=False
    )

    print(f"\n--- Training XGBOOST | Feature: {feature_subset} | Test Fold: {test_fold} ---")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=20
    )

    checkpoint_path = CHECKPOINTS_DIR / f"xgboost_{feature_subset}_fold{test_fold}_best.json"
    model.save_model(str(checkpoint_path))

    pred_probs, pred_classes = evaluate_xgboost_model(model, X_test)
    metrics = compute_metrics(y_test_raw, pred_probs, class_names=LABEL_NAMES)
    metrics["test_fold"] = test_fold
    metrics["model"] = "xgboost"
    metrics["feature_subset"] = feature_subset
    metrics["checkpoint_path"] = str(checkpoint_path)

    cm_fig_path = FIGURES_DIR / f"xgboost_{feature_subset}_fold{test_fold}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=LABEL_NAMES,
        title=f"XGBoost - Fold {test_fold} (Acc: {metrics['accuracy']*100:.2f}%)",
        save_path=cm_fig_path
    )

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("UTA-RLDD DROWSINESS DETECTION TRAINING PIPELINE")
    print("=" * 70)
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Feature Subset     : {FEATURE_SUBSETS[args.features]['name']}")
    print(f"Dataset File       : {args.data_path}")
    print(f"Random Seed        : {args.seed}")
    print("=" * 70)

    # 1. Load dataset
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Dataset file not found at: {args.data_path}")
        print("Please run 'extract_features.py' first or specify a valid --data_path.")
        sys.exit(1)

    print("Loading dataset...")
    dataset = UTARLDDDataset.load_from_files(args.data_path)
    print(f"Dataset successfully loaded: {len(dataset)} sequence samples.")

    # 2. Determine folds to run
    folds_to_run = [args.fold] if args.fold in range(1, 6) else list(range(1, TOTAL_FOLDS + 1))

    fold_results = []
    for f_idx in folds_to_run:
        if args.model == "xgboost":
            res = train_single_fold_xgboost(
                feature_subset=args.features,
                test_fold=f_idx,
                dataset=dataset,
                seed=args.seed
            )
        else:
            res = train_single_fold_nn(
                model_name=args.model,
                feature_subset=args.features,
                test_fold=f_idx,
                dataset=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed
            )

        fold_results.append(res)
        print(f"\nFold {f_idx} Results -> Accuracy: {res['accuracy']*100:.2f}%, Macro F1: {res['f1_macro']*100:.2f}%")

    # 3. Compute 5-Fold Cross Validation Average & Std
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    precs = [r["precision_macro"] for r in fold_results]
    recs = [r["recall_macro"] for r in fold_results]

    summary = {
        "model": args.model,
        "feature_subset": args.features,
        "num_folds_evaluated": len(fold_results),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "mean_precision_macro": float(np.mean(precs)),
        "std_precision_macro": float(np.std(precs)),
        "mean_recall_macro": float(np.mean(recs)),
        "std_recall_macro": float(np.std(recs)),
        "folds": fold_results
    }

    # Save summary metrics
    summary_path = METRICS_DIR / f"{args.model}_{args.features}_cv_summary.json"
    save_json(summary, summary_path)

    print("\n" + "=" * 70)
    print(f"5-FOLD CROSS-VALIDATION SUMMARY: {args.model.upper()} ({args.features})")
    print("=" * 70)
    for r in fold_results:
        print(f"  Fold {r['test_fold']}: Accuracy = {r['accuracy']*100:6.2f}% | F1-Score = {r['f1_macro']*100:6.2f}%")
    print("-" * 70)
    print(f"  Mean Accuracy  : {summary['mean_accuracy']*100:.2f}% ± {summary['std_accuracy']*100:.2f}%")
    print(f"  Mean Macro F1  : {summary['mean_f1_macro']*100:.2f}% ± {summary['std_f1_macro']*100:.2f}%")
    print(f"  Mean Precision : {summary['mean_precision_macro']*100:.2f}% ± {summary['std_precision_macro']*100:.2f}%")
    print(f"  Mean Recall    : {summary['mean_recall_macro']*100:.2f}% ± {summary['std_recall_macro']*100:.2f}%")
    print(f"\nSaved CV summary report to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
