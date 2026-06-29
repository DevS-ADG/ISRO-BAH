"""
ASTRA Training — Random Forest Phase 1 Binary Classifier Training.

Trains the RF classifier on curated label set. Handles class imbalance
with balanced class weights. Evaluates with ROC AUC, precision, recall, F1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from astra.classification.rf_classifier import RFClassifier
from astra.extraction.feature_extractor import FEATURE_NAMES
from astra.utils.logger import get_logger

logger = get_logger("astra.training.train_rf")


def train_rf_classifier(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    model_dir: str = "models/rf_phase1/",
    report_dir: str = "models/training_reports/",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    n_estimators: int = 500,
    random_seed: int = 42,
) -> dict:
    """Train and evaluate the RF Phase 1 binary classifier.

    Args:
        features_df: DataFrame with 19 feature columns.
        labels: Binary labels (0=NOISE, 1=SIGNAL).
        model_dir: Directory to save the trained model.
        report_dir: Directory for training reports.
        train_fraction: Fraction for training set.
        val_fraction: Fraction for validation set.
        test_fraction: Fraction for test set.
        n_estimators: Number of trees.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Training RF classifier: {len(labels)} samples")

    # Extract feature matrix
    X = features_df[FEATURE_NAMES].values.astype(np.float64)
    y = labels.astype(int)

    # Split: train / val+test, then val / test
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
    clf = RFClassifier(model_dir=model_dir, n_estimators=n_estimators)
    clf.build()
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    val_labels, val_proba = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_labels)
    val_auc = roc_auc_score(y_val, val_proba)

    logger.info(f"Validation accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

    # Evaluate on test set
    test_labels, test_proba = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_labels)
    test_auc = roc_auc_score(y_test, test_proba)
    test_report = classification_report(y_test, test_labels, target_names=["NOISE", "SIGNAL"])
    test_cm = confusion_matrix(y_test, test_labels)

    logger.info(f"Test accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
    logger.info(f"\n{test_report}")

    # Save model
    clf.save()

    # Save training report
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / "rf_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASTRA — Random Forest Phase 1 Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Validation samples: {len(y_val)}\n")
        f.write(f"Test samples: {len(y_test)}\n\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n\n")
        f.write("Classification Report (Test Set):\n")
        f.write(test_report + "\n\n")
        f.write(f"Confusion Matrix:\n{test_cm}\n\n")
        if clf.feature_importances_ is not None:
            f.write("Feature Importances:\n")
            sorted_idx = np.argsort(clf.feature_importances_)[::-1]
            for i in sorted_idx:
                if i < len(FEATURE_NAMES):
                    f.write(f"  {FEATURE_NAMES[i]:25s} {clf.feature_importances_[i]:.4f}\n")

    logger.info(f"Training report saved to {report_path}")

    return {
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "test_confusion_matrix": test_cm,
    }
