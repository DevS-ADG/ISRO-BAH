"""
ASTRA Feature Extractor — Compute the 19-feature vector for ML classification.

Extracts the complete 19 physics-informed features deterministically from
available data. Missing features are recorded as numpy.nan, NOT zero,
to avoid biasing the classifier.
"""

import numpy as np
from scipy.stats import pearsonr

from astra.utils.logger import get_logger
from astra.utils.stellar_utils import (
    equilibrium_temperature,
    expected_transit_duration_hours,
    impact_parameter_proxy,
    planet_radius_earth,
    semi_major_axis_au,
    semi_major_axis_stellar_radii,
    estimate_stellar_mass,
)

logger = get_logger("astra.extraction.feature_extractor")

# Ordered list of all 19 features
FEATURE_NAMES = [
    "depth",
    "period",
    "duration",
    "snr",
    "n_transit",
    "flat_bottom_ratio",
    "ingress_symmetry",
    "odd_even_sigma",
    "secondary_depth_ratio",
    "impact_param_proxy",
    "chi2_ratio",
    "oot_rms",
    "centroid_shift",
    "r_star",
    "t_eff",
    "crowdsap",
    "r_planet_earth",
    "t_eq",
    "duration_ratio",
]


def extract_features(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    depth: float,
    snr: float,
    n_transit: int,
    r_star: float,
    teff: float,
    crowdsap: float,
    transit_times: np.ndarray | None = None,
    time: np.ndarray | None = None,
    raw_flux: np.ndarray | None = None,
    centroid_shift_value: float = np.nan,
    odd_even_sigma_value: float = np.nan,
    secondary_depth_ratio_value: float = np.nan,
    flat_bottom_ratio_value: float = np.nan,
    ingress_symmetry_value: float = np.nan,
    duration_ratio_value: float = np.nan,
) -> dict[str, float]:
    """Extract the complete 19-feature vector for a transit candidate.

    All features are computed as float64. Missing or uncomputable features
    are recorded as numpy.nan, NOT zero, to avoid biasing the classifier.

    Args:
        phase: Unbinned phase-folded phase array.
        flux: Unbinned phase-folded flux array.
        flux_err: Flux uncertainty array.
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        period: Orbital period in days.
        t0: Transit mid-time.
        duration_days: Transit duration in days.
        depth: Fractional transit depth.
        snr: Signal-to-noise ratio.
        n_transit: Number of transit events observed.
        r_star: Stellar radius in solar radii.
        teff: Effective temperature in Kelvin.
        crowdsap: CROWDSAP contamination metric.
        transit_times: Array of individual transit mid-times (for odd/even).
        time: Full time array (for odd/even computation).
        raw_flux: Full flux array (for odd/even computation).
        centroid_shift_value: Pre-computed centroid shift (from vetting).
        odd_even_sigma_value: Pre-computed odd/even sigma (from vetting).
        secondary_depth_ratio_value: Pre-computed secondary depth ratio.
        flat_bottom_ratio_value: Pre-computed flat bottom ratio.
        ingress_symmetry_value: Pre-computed ingress symmetry.
        duration_ratio_value: Pre-computed duration ratio.

    Returns:
        Dictionary with exactly 19 named features (values as float64).
    """
    features: dict[str, float] = {}

    # ── 1. depth ────────────────────────────────────────────────────────
    # Transit depth: mean in-transit flux subtracted from 1.0
    features["depth"] = float(depth) if np.isfinite(depth) else np.nan

    # ── 2. period ───────────────────────────────────────────────────────
    features["period"] = float(period) if np.isfinite(period) else np.nan

    # ── 3. duration ─────────────────────────────────────────────────────
    # Transit duration in hours
    duration_hours = duration_days * 24.0 if np.isfinite(duration_days) else np.nan
    features["duration"] = float(duration_hours)

    # ── 4. snr ──────────────────────────────────────────────────────────
    features["snr"] = float(snr) if np.isfinite(snr) else np.nan

    # ── 5. n_transit ────────────────────────────────────────────────────
    features["n_transit"] = float(n_transit)

    # ── 6. flat_bottom_ratio ────────────────────────────────────────────
    if np.isfinite(flat_bottom_ratio_value):
        features["flat_bottom_ratio"] = float(flat_bottom_ratio_value)
    else:
        features["flat_bottom_ratio"] = _compute_flat_bottom_ratio(
            binned_phase, binned_flux, period, duration_days
        )

    # ── 7. ingress_symmetry ─────────────────────────────────────────────
    if np.isfinite(ingress_symmetry_value):
        features["ingress_symmetry"] = float(ingress_symmetry_value)
    else:
        features["ingress_symmetry"] = _compute_ingress_symmetry(
            binned_phase, binned_flux, period, duration_days
        )

    # ── 8. odd_even_sigma ───────────────────────────────────────────────
    features["odd_even_sigma"] = float(odd_even_sigma_value)

    # ── 9. secondary_depth_ratio ────────────────────────────────────────
    features["secondary_depth_ratio"] = float(secondary_depth_ratio_value)

    # ── 10. impact_param_proxy ──────────────────────────────────────────
    m_star = estimate_stellar_mass(teff, r_star)
    features["impact_param_proxy"] = impact_parameter_proxy(
        duration_hours, period, r_star, m_star
    )

    # ── 11. chi2_ratio ──────────────────────────────────────────────────
    features["chi2_ratio"] = _compute_chi2_ratio(
        binned_phase, binned_flux, depth, period, duration_days
    )

    # ── 12. oot_rms ─────────────────────────────────────────────────────
    features["oot_rms"] = _compute_oot_rms(phase, flux, period, duration_days)

    # ── 13. centroid_shift ──────────────────────────────────────────────
    features["centroid_shift"] = float(centroid_shift_value)

    # ── 14. r_star ──────────────────────────────────────────────────────
    features["r_star"] = float(r_star) if np.isfinite(r_star) else np.nan

    # ── 15. t_eff ───────────────────────────────────────────────────────
    features["t_eff"] = float(teff) if np.isfinite(teff) else np.nan

    # ── 16. crowdsap ────────────────────────────────────────────────────
    features["crowdsap"] = float(crowdsap) if np.isfinite(crowdsap) else np.nan

    # ── 17. r_planet_earth ──────────────────────────────────────────────
    features["r_planet_earth"] = planet_radius_earth(depth, r_star)

    # ── 18. t_eq ────────────────────────────────────────────────────────
    a_au = semi_major_axis_au(period, m_star)
    features["t_eq"] = equilibrium_temperature(teff, r_star, a_au)

    # ── 19. duration_ratio ──────────────────────────────────────────────
    if np.isfinite(duration_ratio_value):
        features["duration_ratio"] = float(duration_ratio_value)
    else:
        a_rs = semi_major_axis_stellar_radii(period, m_star, r_star)
        expected_dur = expected_transit_duration_hours(period, r_star, a_rs)
        if expected_dur > 0 and np.isfinite(expected_dur):
            features["duration_ratio"] = duration_hours / expected_dur
        else:
            features["duration_ratio"] = np.nan

    # Validate all features are float64
    for key in FEATURE_NAMES:
        if key not in features:
            features[key] = np.nan
            logger.warning(f"Feature '{key}' not computed, set to NaN")
        features[key] = np.float64(features[key])

    return features


def _compute_flat_bottom_ratio(
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    period: float,
    duration_days: float,
) -> float:
    """Compute flat bottom ratio from phase-folded binned data.

    Fits a trapezoid and computes W_flat / W_total.

    Args:
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        period: Period in days.
        duration_days: Duration in days.

    Returns:
        Flat bottom ratio (0 to 1). Near 0 = V-shape, near 1 = flat bottom.
    """
    if len(binned_phase) < 10 or np.isnan(period) or np.isnan(duration_days):
        return np.nan

    try:
        half_dur_phase = (duration_days / period) / 2.0
        in_transit = np.abs(binned_phase) <= half_dur_phase

        if np.sum(in_transit) < 5:
            return np.nan

        transit_flux = binned_flux[in_transit]
        transit_phase = binned_phase[in_transit]

        # Find the bottom (minimum flux region)
        min_flux = np.min(transit_flux)
        max_flux = np.max(transit_flux)
        flux_range = max_flux - min_flux

        if flux_range <= 0:
            return np.nan

        # "Flat bottom" = points within 10% of the minimum
        flat_threshold = min_flux + 0.1 * flux_range
        flat_mask = transit_flux <= flat_threshold

        if np.sum(flat_mask) < 2:
            return 0.0

        flat_width = np.ptp(transit_phase[flat_mask])
        total_width = np.ptp(transit_phase)

        if total_width <= 0:
            return np.nan

        return float(np.clip(flat_width / total_width, 0.0, 1.0))

    except Exception:
        return np.nan


def _compute_ingress_symmetry(
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    period: float,
    duration_days: float,
) -> float:
    """Compute ingress-egress symmetry via Pearson correlation.

    Splits transit into ingress and egress, flips egress, and computes
    correlation. Values near 1.0 indicate symmetric transits.

    Args:
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        period: Period in days.
        duration_days: Duration in days.

    Returns:
        Pearson correlation coefficient (−1 to 1).
    """
    if len(binned_phase) < 10 or np.isnan(period) or np.isnan(duration_days):
        return np.nan

    try:
        half_dur_phase = (duration_days / period) / 2.0

        # Ingress: negative phase side of transit
        ingress_mask = (binned_phase >= -half_dur_phase) & (binned_phase < 0)
        egress_mask = (binned_phase > 0) & (binned_phase <= half_dur_phase)

        ingress_flux = binned_flux[ingress_mask]
        egress_flux = binned_flux[egress_mask]

        if len(ingress_flux) < 3 or len(egress_flux) < 3:
            return np.nan

        # Resample to same length
        min_len = min(len(ingress_flux), len(egress_flux))
        ingress_resample = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(ingress_flux)),
            ingress_flux,
        )
        egress_resample = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(egress_flux)),
            egress_flux[::-1],  # Flip egress for comparison
        )

        corr, _ = pearsonr(ingress_resample, egress_resample)
        return float(corr) if np.isfinite(corr) else np.nan

    except Exception:
        return np.nan


def _compute_chi2_ratio(
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    depth: float,
    period: float,
    duration_days: float,
) -> float:
    """Compute χ² ratio: flat line fit vs transit model fit.

    chi2_ratio = χ²_flat / χ²_transit

    Higher values indicate a better transit model fit.

    Args:
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        depth: Transit depth.
        period: Period in days.
        duration_days: Duration in days.

    Returns:
        Chi-squared ratio (>1 means transit model is better than flat).
    """
    if len(binned_flux) < 5 or np.isnan(depth):
        return np.nan

    try:
        # Flat model: constant at median
        flat_model = np.nanmedian(binned_flux)
        chi2_flat = np.nansum((binned_flux - flat_model) ** 2)

        # Simple box transit model
        half_dur_phase = (duration_days / period) / 2.0 if period > 0 else 0.1
        in_transit = np.abs(binned_phase) <= half_dur_phase

        transit_model = np.ones_like(binned_flux) * flat_model
        transit_model[in_transit] = flat_model - depth

        chi2_transit = np.nansum((binned_flux - transit_model) ** 2)

        if chi2_transit <= 0:
            return np.nan

        return float(chi2_flat / chi2_transit)

    except Exception:
        return np.nan


def _compute_oot_rms(
    phase: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration_days: float,
) -> float:
    """Compute RMS of out-of-transit flux.

    Measures stellar variability and noise level.

    Args:
        phase: Phase array.
        flux: Phase-folded flux array.
        period: Period in days.
        duration_days: Duration in days.

    Returns:
        RMS of out-of-transit flux.
    """
    if len(phase) < 10:
        return np.nan

    try:
        half_dur_phase = (duration_days / period) / 2.0 if period > 0 else 0.1
        oot_mask = np.abs(phase) > half_dur_phase * 1.5  # Add buffer

        oot_flux = flux[oot_mask]

        if len(oot_flux) < 5:
            return np.nan

        return float(np.nanstd(oot_flux))

    except Exception:
        return np.nan


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert feature dictionary to ordered numpy array.

    Args:
        features: Feature dictionary with the 19 named features.

    Returns:
        Float64 array of shape (19,) in canonical feature order.
    """
    return np.array(
        [features.get(name, np.nan) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def array_to_features(arr: np.ndarray) -> dict[str, float]:
    """Convert ordered numpy array back to feature dictionary.

    Args:
        arr: Float64 array of shape (19,).

    Returns:
        Feature dictionary with the 19 named features.
    """
    return {name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)}
