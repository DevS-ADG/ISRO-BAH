"""
ASTRA XGBoost Classifier — Phase 2 Multi-Class Classification.

Classifies passing signals into 4 astrophysically meaningful classes:
PLANET, ECLIPSING_BINARY, BLEND, OTHER.

Uses XGBClassifier with objective='multi:softprob' for probability output.
"""

import numpy as np
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.classification.xgb_classifier")

# Class mapping
CLASS_NAMES = ["PLANET", "ECLIPSING_BINARY", "BLEND", "OTHER"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class XGBMultiClassifier:
    """XGBoost multi-class classifier for transit signal classification.

    Phase 2: Classifies signals into PLANET / EB / BLEND / OTHER with
    full probability vector output.

    Args:
        model_dir: Directory for serialized model.
        n_estimators: Number of boosting rounds (default 300).
        max_depth: Maximum tree depth (default 6).
        learning_rate: Boosting learning rate (default 0.05).
        subsample: Row subsample ratio (default 0.8).
        colsample_bytree: Column subsample ratio (default 0.8).
    """

    def __init__(
        self,
        model_dir: str = "models/xgboost_phase2/",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def build(self) -> None:
        """Build the XGBoost model and scaler."""
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        self.scaler = StandardScaler()
        logger.info("XGBoost classifier built")

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the XGBoost classifier.

        Args:
            X: Feature matrix (n_samples, 19).
            y: Class labels (0=PLANET, 1=EB, 2=BLEND, 3=OTHER).

        Returns:
            Training metrics.
        """
        if self.model is None:
            self.build()

        X_imputed = self._impute_nan(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Dynamically set num_class and objective based on unique classes in y
        n_classes = len(set(y.tolist()))
        if n_classes == 2:
            self.model.set_params(objective="binary:logistic")
        else:
            self.model.set_params(objective="multi:softprob", num_class=n_classes)

        # Compute sample weights for class imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y)

        self.model.fit(X_scaled, y, sample_weight=sample_weights)

        # Metrics
        from sklearn.metrics import classification_report, accuracy_score

        y_pred = self.model.predict(X_scaled)
        # y_pred may be one-hot encoded for multi-class — handle both
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y, y_pred)

        # Build target names only for classes that are present
        present_classes = sorted(set(y.tolist()))
        present_names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in present_classes]
        report = classification_report(y, y_pred, labels=present_classes, target_names=present_names)

        logger.info(f"XGBoost training accuracy: {accuracy:.4f}")
        logger.info(f"\n{report}")

        return {"accuracy": accuracy}

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities for candidates.

        Args:
            X: Feature matrix (n_candidates, 19).

        Returns:
            Tuple of (class_labels, probability_matrix).
            class_labels: Array of class indices (argmax of probabilities).
            probability_matrix: Shape (n_candidates, 4) with probabilities
                for [PLANET, EB, BLEND, OTHER] summing to 1.0.
        """
        if self.model is None:
            raise RuntimeError("XGBoost model not trained. Call fit() or load() first.")

        X_imputed = self._impute_nan(X)
        X_scaled = self.scaler.transform(X_imputed)

        proba = self.model.predict_proba(X_scaled)
        labels = np.argmax(proba, axis=1)

        return labels, proba

    def predict_single(self, features: np.ndarray) -> tuple[str, np.ndarray]:
        """Predict class for a single candidate.

        Args:
            features: Feature vector of shape (19,).

        Returns:
            Tuple of (class_name, probability_vector).
        """
        X = features.reshape(1, -1)
        labels, proba = self.predict(X)
        class_name = CLASS_NAMES[labels[0]]
        return class_name, proba[0]

    def save(self) -> None:
        """Save trained model and scaler."""
        import joblib

        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_dir / "xgb_model.joblib")
        joblib.dump(self.scaler, self.model_dir / "xgb_scaler.joblib")
        logger.info(f"XGBoost model saved to {self.model_dir}")

    def load(self) -> bool:
        """Load a previously trained model.

        Returns:
            True if loaded successfully.
        """
        import joblib

        model_path = self.model_dir / "xgb_model.joblib"
        scaler_path = self.model_dir / "xgb_scaler.joblib"

        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"XGBoost model loaded from {self.model_dir}")
            return True

        logger.warning(f"XGBoost model not found at {self.model_dir}")
        return False

    @staticmethod
    def _impute_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN with column medians."""
        X_copy = X.copy()
        for col in range(X_copy.shape[1]):
            mask = np.isnan(X_copy[:, col])
            if np.any(mask):
                median_val = np.nanmedian(X_copy[:, col])
                X_copy[mask, col] = median_val if np.isfinite(median_val) else 0.0
        return X_copy
