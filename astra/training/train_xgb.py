"""
ASTRA Training — XGBoost Phase 2 Multi-Class Classifier Training.

Trains the XGBoost multi-class classifier on the curated label set.
Handles class imbalance via balanced sample weights.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from astra.classification.xgb_classifier import XGBMultiClassifier, CLASS_NAMES
from astra.extraction.feature_extractor import FEATURE_NAMES
from astra.utils.logger import get_logger

logger = get_logger("astra.training.train_xgb")


def train_xgb_classifier(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    model_dir: str = "models/xgboost_phase2/",
    report_dir: str = "models/training_reports/",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
    **xgb_kwargs,
) -> dict:
    """Train and evaluate the XGBoost Phase 2 multi-class classifier.

    Args:
        features_df: DataFrame with 19 feature columns.
        labels: Multi-class labels (0=PLANET, 1=EB, 2=BLEND, 3=OTHER).
        model_dir: Directory to save the trained model.
        report_dir: Directory for training reports.
        train_fraction: Training set fraction.
        val_fraction: Validation set fraction.
        test_fraction: Test set fraction.
        random_seed: Random seed.
        **xgb_kwargs: Additional XGBoost hyperparameters.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Training XGBoost classifier: {len(labels)} samples")

    X = features_df[FEATURE_NAMES].values.astype(np.float64)
    y = labels.astype(int)

    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_fraction + test_fraction),
        stratify=y, random_state=random_seed,
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        stratify=y_temp, random_state=random_seed,
    )

    logger.info(
        f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )

    # Build and train
    clf = XGBMultiClassifier(model_dir=model_dir, **xgb_kwargs)
    clf.build()
    clf.fit(X_train, y_train)

    # Validation evaluation
    val_labels, val_proba = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_labels)

    try:
        val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr", average="macro")
    except ValueError:
        val_auc = np.nan

    logger.info(f"Validation accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

    # Test evaluation
    test_labels, test_proba = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_labels)

    try:
        test_auc = roc_auc_score(y_test, test_proba, multi_class="ovr", average="macro")
    except ValueError:
        test_auc = np.nan

    # Use only the classes that appear in the test set
    present_classes = sorted(set(y_test.tolist()))
    present_names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in present_classes]
    test_report = classification_report(y_test, test_labels, labels=present_classes, target_names=present_names)
    test_cm = confusion_matrix(y_test, test_labels)

    logger.info(f"Test accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
    logger.info(f"\n{test_report}")

    # Save model
    clf.save()

    # Save report
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / "xgb_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASTRA — XGBoost Phase 2 Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classes: {CLASS_NAMES}\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Validation: accuracy={val_accuracy:.4f}, AUC={val_auc:.4f}\n")
        f.write(f"Test: accuracy={test_accuracy:.4f}, AUC={test_auc:.4f}\n\n")
        f.write("Classification Report (Test Set):\n")
        f.write(test_report + "\n\n")
        f.write(f"Confusion Matrix:\n{test_cm}\n")

    logger.info(f"Training report saved to {report_path}")

    return {
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "test_confusion_matrix": test_cm,
    }
