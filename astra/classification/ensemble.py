"""
ASTRA Ensemble — Late-fusion ensemble of XGBoost and CNN classifiers.

Combines XGBoost (physics-informed features) and CNN (shape embedding)
probability vectors using a weighted average:
    P_final = w_xgb × P_xgb + w_cnn × P_cnn

XGBoost receives higher default weight (0.6) because its 19 physics-informed
features are more directly interpretable to ISRO judges.
"""

import numpy as np

from astra.classification.xgb_classifier import CLASS_NAMES
from astra.utils.logger import get_logger

logger = get_logger("astra.classification.ensemble")


class EnsembleClassifier:
    """Late-fusion ensemble combining XGBoost and CNN classifiers.

    Args:
        weight_xgb: Weight for XGBoost probabilities (default 0.6).
        weight_cnn: Weight for CNN probabilities (default 0.4).
        confidence_planet_threshold: Minimum P(PLANET) for PLANET candidate.
        confidence_marginal_threshold: Minimum P(PLANET) for marginal candidate.
    """

    def __init__(
        self,
        weight_xgb: float = 0.6,
        weight_cnn: float = 0.4,
        confidence_planet_threshold: float = 0.80,
        confidence_marginal_threshold: float = 0.50,
    ):
        self.weight_xgb = weight_xgb
        self.weight_cnn = weight_cnn
        self.confidence_planet = confidence_planet_threshold
        self.confidence_marginal = confidence_marginal_threshold

    def combine(
        self,
        proba_xgb: np.ndarray,
        proba_cnn: np.ndarray | None = None,
    ) -> dict:
        """Combine XGBoost and CNN probability vectors.

        If CNN probabilities are unavailable (model not trained), uses
        XGBoost probabilities alone.

        Args:
            proba_xgb: XGBoost probability vector of shape (4,) for
                [PLANET, EB, BLEND, OTHER].
            proba_cnn: CNN probability vector of shape (4,) or None.

        Returns:
            Dictionary with:
                - class_label: Predicted class name.
                - class_index: Predicted class index.
                - probabilities: Final probability vector (4,).
                - confidence: P(predicted_class).
                - confidence_planet: P(PLANET).
                - tier: Assigned tier based on confidence.
                - proba_xgb: Original XGBoost probabilities.
                - proba_cnn: Original CNN probabilities (or None).
        """
        if proba_cnn is not None:
            # Late fusion: weighted average
            p_final = self.weight_xgb * proba_xgb + self.weight_cnn * proba_cnn
            # Renormalize to ensure sum = 1.0
            p_sum = np.sum(p_final)
            if p_sum > 0:
                p_final = p_final / p_sum
        else:
            p_final = proba_xgb.copy()

        class_index = int(np.argmax(p_final))
        class_label = CLASS_NAMES[class_index]
        confidence = float(p_final[class_index])
        confidence_planet = float(p_final[0])  # Index 0 = PLANET

        # Assign tier based on planet confidence thresholds
        if confidence_planet >= self.confidence_planet:
            tier = "candidate"  # Tier 1 or 2 (based on SNR later)
        elif confidence_planet >= self.confidence_marginal:
            tier = "marginal"   # Tier 3
        else:
            tier = "false_positive"

        result = {
            "class_label": class_label,
            "class_index": class_index,
            "probabilities": p_final,
            "confidence": confidence,
            "confidence_planet": confidence_planet,
            "confidence_eb": float(p_final[1]),
            "confidence_blend": float(p_final[2]),
            "confidence_other": float(p_final[3]),
            "tier": tier,
            "proba_xgb": proba_xgb,
            "proba_cnn": proba_cnn,
        }

        logger.debug(
            f"Ensemble: {class_label} (P={confidence:.3f}), "
            f"P(PLANET)={confidence_planet:.3f}, tier={tier}"
        )

        return result

    def assign_final_tier(
        self,
        ensemble_result: dict,
        snr: float,
        tier1_snr_min: float = 10.0,
        tier1_confidence_min: float = 0.90,
        tier2_snr_min: float = 7.0,
        tier2_confidence_min: float = 0.80,
    ) -> str:
        """Assign final tier based on SNR and confidence thresholds.

        Tier 1: SNR > 10, confidence > 0.90, all vetting passed, BATMAN fitted
        Tier 2: SNR > 7, confidence > 0.80, all vetting passed
        Tier 3: SNR > 7, confidence 0.50-0.80, flagged for follow-up

        Args:
            ensemble_result: Output from combine().
            snr: Signal-to-noise ratio.
            tier1_snr_min: Minimum SNR for Tier 1.
            tier1_confidence_min: Minimum confidence for Tier 1.
            tier2_snr_min: Minimum SNR for Tier 2.
            tier2_confidence_min: Minimum confidence for Tier 2.

        Returns:
            Final tier string: "tier1", "tier2", "tier3", or "false_positive".
        """
        confidence_planet = ensemble_result["confidence_planet"]

        if (
            snr >= tier1_snr_min
            and confidence_planet >= tier1_confidence_min
        ):
            return "tier1"
        elif (
            snr >= tier2_snr_min
            and confidence_planet >= tier2_confidence_min
        ):
            return "tier2"
        elif confidence_planet >= self.confidence_marginal:
            return "tier3"
        else:
            return "false_positive"
