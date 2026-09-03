"""
Utility functions for reproducibility, metrics calculation, plotting, and file I/O.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Compatibility patch for older seaborn versions with newer matplotlib (>= 3.9)
if not hasattr(matplotlib.cm, "register_cmap") and hasattr(matplotlib, "colormaps"):
    matplotlib.cm.register_cmap = matplotlib.colormaps.register

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    log_loss
)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, OS, and TensorFlow/PyTorch
    to ensure 100% reproducible experiments.

    Args:
        seed: Integer random seed (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        # Enable deterministic operations if supported
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except ImportError:
        pass


def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict, List]]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: 1D array of ground truth class indices (e.g. [0, 1, 2, ...]).
        y_pred_probs: 2D array of predicted class probabilities or 1D predicted class indices.
        class_names: List of class string names (e.g. ['Alert', 'Low Vigilant', 'Drowsy']).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, log_loss, confusion matrix,
        and per-class classification report.
    """
    if class_names is None:
        class_names = ["Alert", "Low Vigilant", "Drowsy"]

    if y_pred_probs.ndim == 2:
        y_pred = np.argmax(y_pred_probs, axis=1)
        loss_val = float(log_loss(y_true, y_pred_probs, labels=list(range(len(class_names)))))
    else:
        y_pred = y_pred_probs
        loss_val = None

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "log_loss": loss_val,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }
    return metrics


def plot_learning_curves(
    history_dict: Dict[str, List[float]],
    title: str = "Training & Validation Curves",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot training & validation Loss and Accuracy learning curves.

    Args:
        history_dict: Dictionary containing 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        title: Title of the overall plot.
        save_path: Optional path to save high-res figure (e.g. .png).
    """
    plt.figure(figsize=(14, 5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="Train Accuracy", color="#2980b9", linewidth=2)
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Val Accuracy", color="#c0392b", linestyle="--", linewidth=2)
    plt.title(f"{title} - Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    plt.subplot(1, 2, 2)
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], label="Train Loss", color="#2980b9", linewidth=2)
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Val Loss", color="#c0392b", linestyle="--", linewidth=2)
    plt.title(f"{title} - Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = False
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: 2D array or list of lists confusion matrix.
        class_names: List of class labels.
        title: Title of plot.
        save_path: Optional filepath to save the figure.
        normalize: Whether to normalize matrix values to percentages [0, 1].
    """
    cm_arr = np.array(cm, dtype=np.float64)
    fmt = "d"
    if normalize:
        cm_arr = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2%"

    plt.figure(figsize=(7, 6))
    if SEABORN_AVAILABLE:
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 11, "weight": "bold"}
        )
    else:
        im = plt.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=10)
        plt.yticks(tick_marks, class_names, fontsize=10)
        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                val = cm_arr[i, j]
                txt = f"{val:.2%}" if normalize else f"{int(val)}"
                plt.text(
                    j, i, txt,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold"
                )

    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_bar_chart(
    results: Dict[str, Dict[str, float]],
    metric_key: str = "accuracy",
    metric_display_name: str = "Accuracy (%)",
    title: str = "Model Performance Comparison across Feature Sets",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot grouped bar chart comparing multiple models across different feature subsets.

    Args:
        results: Nested dict of format: {model_name: {feature_subset: metric_value_0_to_100, ...}, ...}
        metric_key: Metric key name.
        metric_display_name: Y-axis label.
        title: Chart title.
        save_path: Optional filepath to save figure.
    """
    plt.figure(figsize=(11, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    models = list(results.keys())
    if not models:
        return

    subsets = list(results[models[0]].keys())
    x = np.arange(len(subsets))
    width = 0.8 / len(models)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model_name in enumerate(models):
        scores = [results[model_name].get(s, 0.0) for s in subsets]
        # if scores are in [0, 1], scale to percentage [0, 100]
        if max(scores) <= 1.0 and max(scores) > 0:
            scores = [s * 100 for s in scores]

        offset = (i - (len(models) - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            scores,
            width,
            label=model_name.upper(),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8
        )

        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold"
                )

    ax.set_ylabel(metric_display_name, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_json(data: Union[Dict, List], filepath: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
