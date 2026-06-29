"""
ASTRA RF Classifier — Phase 1 Random Forest Binary Classifier.

First-pass gate: separates genuine astrophysical signals (planet, EB, blend)
from instrumental noise and stellar variability. Binary output: SIGNAL or NOISE.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from astra.extraction.feature_extractor import FEATURE_NAMES
from astra.utils.logger import get_logger

logger = get_logger("astra.classification.rf_classifier")


class RFClassifier:
    """Random Forest binary classifier for signal vs noise separation.

    Phase 1 of the classification pipeline. Candidates classified as NOISE
    with probability > 0.8 are discarded from further processing.

    Args:
        model_dir: Directory containing the serialized model and scaler.
        n_estimators: Number of trees (default 500).
        min_samples_split: Minimum samples to split a node (default 5).
    """

    def __init__(
        self,
        model_dir: str = "models/rf_phase1/",
        n_estimators: int = 500,
        min_samples_split: int = 5,
    ):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.feature_importances_: np.ndarray | None = None

    def build(self) -> None:
        """Build the Random Forest model and StandardScaler."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,  # Fully grown trees
            min_samples_split=self.min_samples_split,
            class_weight="balanced",  # Handle class imbalance
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        logger.info("RF classifier built")

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the RF classifier.

        Args:
            X: Feature matrix of shape (n_samples, 19).
            y: Binary labels (0=NOISE, 1=SIGNAL).

        Returns:
            Training metrics dictionary.
        """
        if self.model is None:
            self.build()

        # Handle NaN values — RF can handle them natively via sklearn
        # but scaler cannot. Impute with median for scaling.
        X_imputed = self._impute_nan(X)

        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Train
        self.model.fit(X_scaled, y)

        # Feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Log top 5 features
        top_indices = np.argsort(self.feature_importances_)[::-1][:5]
        top_features = [
            (FEATURE_NAMES[i], self.feature_importances_[i])
            for i in top_indices
            if i < len(FEATURE_NAMES)
        ]
        logger.info(f"Top 5 features: {top_features}")

        # Training accuracy
        from sklearn.metrics import accuracy_score, classification_report

        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=["NOISE", "SIGNAL"])

        logger.info(f"RF training accuracy: {accuracy:.4f}")
        logger.info(f"\n{report}")

        return {"accuracy": accuracy, "top_features": top_features}

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict signal vs noise for candidates.

        Args:
            X: Feature matrix of shape (n_candidates, 19).

        Returns:
            Tuple of (labels, probabilities).
            labels: Binary array (0=NOISE, 1=SIGNAL).
            probabilities: P(SIGNAL) for each candidate.
        """
        if self.model is None:
            raise RuntimeError("RF model not trained. Call fit() or load() first.")

        X_imputed = self._impute_nan(X)
        X_scaled = self.scaler.transform(X_imputed)

        labels = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)

        # P(SIGNAL) is the probability of class 1
        p_signal = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        return labels, p_signal

    def save(self) -> None:
        """Save the trained model and scaler to disk."""
        import joblib

        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_dir / "rf_model.joblib")
        joblib.dump(self.scaler, self.model_dir / "rf_scaler.joblib")
        logger.info(f"RF model saved to {self.model_dir}")

    def load(self) -> bool:
        """Load a previously trained model and scaler.

        Returns:
            True if loaded successfully, False otherwise.
        """
        import joblib

        model_path = self.model_dir / "rf_model.joblib"
        scaler_path = self.model_dir / "rf_scaler.joblib"

        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if hasattr(self.model, "feature_importances_"):
                self.feature_importances_ = self.model.feature_importances_
            logger.info(f"RF model loaded from {self.model_dir}")
            return True

        logger.warning(f"RF model not found at {self.model_dir}")
        return False

    @staticmethod
    def _impute_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN values with column medians.

        Args:
            X: Feature matrix possibly containing NaN.

        Returns:
            Imputed feature matrix.
        """
        X_copy = X.copy()
        for col in range(X_copy.shape[1]):
            mask = np.isnan(X_copy[:, col])
            if np.any(mask):
                median_val = np.nanmedian(X_copy[:, col])
                X_copy[mask, col] = median_val if np.isfinite(median_val) else 0.0
        return X_copy
