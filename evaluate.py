"""
Model Evaluation and Paper-Ready Benchmark Comparison Script.

Reads cross-validation summaries from `output/metrics/`, formats comparison
tables (Markdown & LaTeX), and generates publication-grade comparison figures.
"""

import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import METRICS_DIR, FIGURES_DIR, FEATURE_SUBSETS, SUPPORTED_MODELS
from src.utils import load_json, plot_comparison_bar_chart


def collect_metrics_summaries() -> pd.DataFrame:
    """Collect all *_cv_summary.json files from output/metrics."""
    json_files = glob.glob(str(METRICS_DIR / "*_cv_summary.json"))
    records = []

    for jf in json_files:
        try:
            data = load_json(jf)
            records.append({
                "model": data.get("model", "").upper(),
                "feature_subset": data.get("feature_subset", ""),
                "mean_accuracy": data.get("mean_accuracy", 0.0) * 100,
                "std_accuracy": data.get("std_accuracy", 0.0) * 100,
                "mean_f1_macro": data.get("mean_f1_macro", 0.0) * 100,
                "std_f1_macro": data.get("std_f1_macro", 0.0) * 100,
                "mean_precision_macro": data.get("mean_precision_macro", 0.0) * 100,
                "std_precision_macro": data.get("std_precision_macro", 0.0) * 100,
                "mean_recall_macro": data.get("mean_recall_macro", 0.0) * 100,
                "std_recall_macro": data.get("std_recall_macro", 0.0) * 100,
                "num_folds": data.get("num_folds_evaluated", 0)
            })
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(by=["model", "feature_subset"])
    return df


def generate_comparison_plots(df: pd.DataFrame) -> None:
    """Generate comparative grouped bar charts across models and feature sets."""
    if df.empty:
        return

    # Pivot table for accuracy
    pivot_acc = df.pivot(index="model", columns="feature_subset", values="mean_accuracy")
    results_acc = pivot_acc.to_dict(orient="index")

    chart_path_acc = FIGURES_DIR / "model_feature_accuracy_comparison.png"
    plot_comparison_bar_chart(
        results=results_acc,
        metric_key="accuracy",
        metric_display_name="5-Fold Cross-Validation Accuracy (%)",
        title="UTA-RLDD Drowsiness Detection Accuracy (Ablation Study)",
        save_path=chart_path_acc
    )
    print(f"✅ Accuracy comparison figure saved to: {chart_path_acc}")

    # Pivot table for F1
    pivot_f1 = df.pivot(index="model", columns="feature_subset", values="mean_f1_macro")
    results_f1 = pivot_f1.to_dict(orient="index")

    chart_path_f1 = FIGURES_DIR / "model_feature_f1_comparison.png"
    plot_comparison_bar_chart(
        results=results_f1,
        metric_key="f1",
        metric_display_name="5-Fold Macro F1-Score (%)",
        title="UTA-RLDD Drowsiness Detection Macro F1-Score (Ablation Study)",
        save_path=chart_path_f1
    )
    print(f"✅ F1-Score comparison figure saved to: {chart_path_f1}")


def main():
    print("=" * 70)
    print("UTA-RLDD BENCHMARK EVALUATION & COMPARISON")
    print("=" * 70)

    df = collect_metrics_summaries()
    if df.empty:
        print("No evaluation summary files found in output/metrics/.")
        print("Run 'python train.py' or 'python run_all_experiments.py' first.")
        return

    print("\n--- 5-Fold Cross-Validation Summary Table ---")
    print(df.to_string(index=False))

    # Format paper-ready markdown table
    md_lines = [
        "| Model | Feature Set | Accuracy (%) | Macro F1 (%) | Precision (%) | Recall (%) | Folds |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|"
    ]
    for _, row in df.iterrows():
        feat_name = FEATURE_SUBSETS.get(row["feature_subset"], {}).get("name", row["feature_subset"])
        acc_str = f"{row['mean_accuracy']:.2f} ± {row['std_accuracy']:.2f}"
        f1_str = f"{row['mean_f1_macro']:.2f} ± {row['std_f1_macro']:.2f}"
        prec_str = f"{row['mean_precision_macro']:.2f} ± {row['std_precision_macro']:.2f}"
        rec_str = f"{row['mean_recall_macro']:.2f} ± {row['std_recall_macro']:.2f}"
        md_lines.append(
            f"| **{row['model']}** | {feat_name} | {acc_str} | {f1_str} | {prec_str} | {rec_str} | {int(row['num_folds'])} |"
        )

    md_table = "\n".join(md_lines)
    print("\n--- Paper Ready Markdown Table ---")
    print(md_table)

    # Save to file
    table_path = METRICS_DIR / "benchmark_comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# UTA-RLDD 5-Fold Cross-Validation Benchmark Results\n\n")
        f.write(md_table + "\n")
    print(f"\nSaved Markdown table to: {table_path}")

    # Generate figures
    generate_comparison_plots(df)
    print("=" * 70)


if __name__ == "__main__":
    main()
