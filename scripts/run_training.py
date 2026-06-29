"""
ASTRA Training CLI — Train all ML models.

Usage:
    python run_training.py --config config/config.yaml --labels data/labels/labels.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="ASTRA Model Training",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--labels", type=str, default="data/labels/labels.csv",
        help="Path to curated label CSV file",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "rf", "xgb", "cnn"],
        help="Which model to train (default: all)",
    )

    args = parser.parse_args()

    import yaml
    import numpy as np
    import pandas as pd

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from astra.utils.logger import setup_logging
    setup_logging()

    # Load labels
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        print("Please create a curated label set at data/labels/labels.csv")
        print("Required columns: TIC_ID, candidate_number, label")
        print("Labels: PLANET, ECLIPSING_BINARY, BLEND, OTHER, NOISE")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    print(f"Loaded {len(labels_df)} labelled examples")

    # Load corresponding feature vectors
    features_path = Path("data/candidates/features.csv")
    if not features_path.exists():
        print("Error: Feature vectors not found. Run the pipeline first to generate features.")
        sys.exit(1)

    features_df = pd.read_csv(features_path)

    # Merge labels with features
    merged = pd.merge(labels_df, features_df, on=["TIC_ID", "candidate_number"])

    # Binary labels for RF
    label_map_binary = {
        "PLANET": 1, "ECLIPSING_BINARY": 1, "BLEND": 1, "OTHER": 1, "NOISE": 0,
    }
    binary_labels = merged["label"].map(label_map_binary).values

    # Multi-class labels for XGBoost/CNN
    label_map_multi = {
        "PLANET": 0, "ECLIPSING_BINARY": 1, "BLEND": 2, "OTHER": 3,
    }
    # Filter out NOISE for multi-class training
    signal_mask = merged["label"] != "NOISE"
    multi_features = merged[signal_mask]
    multi_labels = multi_features["label"].map(label_map_multi).values

    train_cfg = config.get("training", {})

    if args.model in ("all", "rf"):
        print("\n" + "=" * 60)
        print("  Training Random Forest (Phase 1)")
        print("=" * 60)
        from astra.training.train_rf import train_rf_classifier
        train_rf_classifier(
            merged, binary_labels,
            train_fraction=train_cfg.get("train_fraction", 0.70),
            val_fraction=train_cfg.get("val_fraction", 0.15),
            test_fraction=train_cfg.get("test_fraction", 0.15),
        )

    if args.model in ("all", "xgb"):
        print("\n" + "=" * 60)
        print("  Training XGBoost (Phase 2)")
        print("=" * 60)
        from astra.training.train_xgb import train_xgb_classifier
        train_xgb_classifier(
            multi_features, multi_labels,
            train_fraction=train_cfg.get("train_fraction", 0.70),
            val_fraction=train_cfg.get("val_fraction", 0.15),
            test_fraction=train_cfg.get("test_fraction", 0.15),
        )

    if args.model in ("all", "cnn"):
        print("\n" + "=" * 60)
        print("  Training CNN (Phase 2)")
        print("=" * 60)
        # CNN needs phase-folded light curve arrays
        curves_path = Path("data/candidates/phase_folded_curves.npy")
        if curves_path.exists():
            X_curves = np.load(curves_path)
            from astra.training.train_cnn import train_cnn_classifier
            train_cnn_classifier(
                X_curves[signal_mask.values], multi_labels,
                use_smote=train_cfg.get("use_smote", True),
            )
        else:
            print("Warning: Phase-folded curves not found. Skipping CNN training.")
            print("Run the pipeline first to generate phase-folded curves at:")
            print(f"  {curves_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
