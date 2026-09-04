"""
Automated Experiment Runner for UTA-RLDD Drowsiness Detection.

Executes the complete experimental grid across all model architectures
and feature subsets, running full 5-fold cross-validation and generating
publication-grade comparative benchmark reports.

Usage:
    python run_all_experiments.py --data_path output/extracted_features/uta_rldd_features_seq30.npz
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SUPPORTED_MODELS, FEATURE_SUBSETS, FEATURES_DIR, SEED, TOTAL_FOLDS
from src.utils import set_seed
from train import train_single_fold_nn, train_single_fold_xgboost
from src.dataset import UTARLDDDataset
from src.utils import save_json
from evaluate import collect_metrics_summaries, generate_comparison_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run end-to-end drowsiness detection experiments across all models and features."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(FEATURES_DIR / "uta_rldd_features_seq30.npz"),
        help="Path to pre-extracted dataset file."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bilstm", "lstm", "bigru", "cnn1d", "transformer", "xgboost"],
        help="List of model architectures to evaluate."
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["ear", "ear_mar", "all"],
        help="List of feature subsets to evaluate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per fold."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("STARTING FULL END-TO-END ABLATION EXPERIMENTS (UTA-RLDD 5-FOLD CV)")
    print("=" * 80)
    print(f"Models   : {args.models}")
    print(f"Features : {args.features}")
    print(f"Dataset  : {args.data_path}")
    print("=" * 80)

    dataset = UTARLDDDataset.load_from_files(args.data_path)
    integrity = dataset.validate()
    folds_to_run = dataset.available_folds
    print(
        f"Integrity check passed: {integrity['num_subjects']} subjects, "
        f"folds {folds_to_run}, padding {integrity['padding_rate']:.2%}."
    )
    for warning in integrity["warnings"]:
        print(f"Warning: {warning}")
    if len(folds_to_run) != TOTAL_FOLDS:
        print(
            f"Warning: this archive contains {len(folds_to_run)} folds, not the complete "
            f"{TOTAL_FOLDS}-fold UTA-RLDD benchmark. Results will be labelled accordingly."
        )
    total_experiments = len(args.models) * len(args.features)
    exp_idx = 0

    for model_name in args.models:
        for feat_sub in args.features:
            exp_idx += 1
            print(f"\n[{exp_idx}/{total_experiments}] RUNNING: Model={model_name.upper()} | Feature={feat_sub}")
            print("-" * 80)

            fold_results = []
            for fold_idx in folds_to_run:
                if model_name == "xgboost":
                    res = train_single_fold_xgboost(
                        feature_subset=feat_sub,
                        test_fold=fold_idx,
                        dataset=dataset,
                        seed=args.seed
                    )
                else:
                    res = train_single_fold_nn(
                        model_name=model_name,
                        feature_subset=feat_sub,
                        test_fold=fold_idx,
                        dataset=dataset,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=1e-3,
                        seed=args.seed
                    )
                fold_results.append(res)

            # Summarize
            import numpy as np
            accs = [r["accuracy"] for r in fold_results]
            f1s = [r["f1_macro"] for r in fold_results]
            precs = [r["precision_macro"] for r in fold_results]
            recs = [r["recall_macro"] for r in fold_results]

            summary = {
                "model": model_name,
                "feature_subset": feat_sub,
                "num_folds_evaluated": len(fold_results),
                "fold_ids": folds_to_run,
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

            from config import METRICS_DIR
            save_json(summary, METRICS_DIR / f"{model_name}_{feat_sub}_cv_summary.json")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED! AGGREGATING FINAL BENCHMARK RESULTS...")
    print("=" * 80)

    df = collect_metrics_summaries()
    print(df.to_string(index=False))
    generate_comparison_plots(df)


if __name__ == "__main__":
    main()
