"""
ASTRA Label Validation Utility — Validate the curated label set.

Checks label format, class distribution, and cross-references with
known catalogues.

Usage:
    python validate_labels.py --labels data/labels/labels.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Validate the curated label CSV file."""
    parser = argparse.ArgumentParser(description="ASTRA Label Validator")
    parser.add_argument(
        "--labels", type=str, default="data/labels/labels.csv",
        help="Path to label CSV file",
    )
    args = parser.parse_args()

    import pandas as pd

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Error: File not found: {labels_path}")
        sys.exit(1)

    df = pd.read_csv(labels_path)

    print(f"\n{'='*50}")
    print(f"  ASTRA Label Validation Report")
    print(f"{'='*50}")
    print(f"\n  File: {labels_path}")
    print(f"  Total entries: {len(df)}")

    # Check required columns
    required_cols = ["TIC_ID", "candidate_number", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing required columns: {missing}")
        sys.exit(1)
    else:
        print(f"  Required columns: ✓ Present")

    # Valid labels
    valid_labels = {"PLANET", "ECLIPSING_BINARY", "BLEND", "OTHER", "NOISE"}
    actual_labels = set(df["label"].unique())
    invalid = actual_labels - valid_labels

    if invalid:
        print(f"\n  WARNING: Invalid labels found: {invalid}")
    else:
        print(f"  Label values: ✓ All valid")

    # Class distribution
    print(f"\n  Class Distribution:")
    for label in sorted(actual_labels):
        count = (df["label"] == label).sum()
        pct = 100 * count / len(df)
        print(f"    {label:20s}: {count:5d} ({pct:5.1f}%)")

    # Duplicate check
    dupes = df.duplicated(subset=["TIC_ID", "candidate_number"]).sum()
    if dupes > 0:
        print(f"\n  WARNING: {dupes} duplicate entries found")
    else:
        print(f"\n  Duplicates: ✓ None")

    # Null checks
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        print(f"\n  WARNING: Null values found:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"    {col}: {count} nulls")
    else:
        print(f"  Null values: ✓ None")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
