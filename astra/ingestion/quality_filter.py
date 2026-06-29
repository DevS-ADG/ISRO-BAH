"""
ASTRA Quality Filter — Pre-filter raw TESS light curves.

Applies quality flag masking, minimum datapoint checks, CROWDSAP
contamination filtering, and stellar activity flagging before any
expensive computation.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.ingestion.quality_filter")


class QualityFilterResult:
    """Result of the quality filtering step for a single star.

    Attributes:
        passed: Whether the star passed all hard filters.
        reason: Reason string if the star was rejected.
        stellar_activity_flag: True if RMS exceeds activity threshold.
        n_valid_points: Number of valid data points after quality masking.
        crowdsap: CROWDSAP value from FITS header.
        raw_rms: RMS of the normalized flux.
        time: Quality-masked time array.
        flux: Quality-masked flux array.
        flux_err: Quality-masked flux error array.
    """

    def __init__(self):
        self.passed: bool = True
        self.reason: str = ""
        self.stellar_activity_flag: bool = False
        self.n_valid_points: int = 0
        self.crowdsap: float = np.nan
        self.raw_rms: float = np.nan
        self.time: np.ndarray | None = None
        self.flux: np.ndarray | None = None
        self.flux_err: np.ndarray | None = None


def apply_quality_filter(
    lc,
    min_datapoints: int = 100,
    min_crowdsap: float = 0.5,
    max_rms_threshold: float = 0.05,
) -> QualityFilterResult:
    """Apply quality filtering to a raw TESS light curve.

    This is a hard pre-filter before any expensive computation.

    Steps:
    1. Mask all cadences with non-zero QUALITY flags (momentum dumps,
       cosmic rays, scattered light, manual excludes).
    2. Check minimum remaining data points after masking.
    3. Check CROWDSAP for contamination.
    4. Compute raw RMS and flag stellar activity.

    Args:
        lc: lightkurve LightCurve object (raw, from FITS).
        min_datapoints: Minimum valid data points required (default 100).
        min_crowdsap: Minimum CROWDSAP value (default 0.5).
        max_rms_threshold: RMS threshold for stellar activity flag (default 0.05).

    Returns:
        QualityFilterResult with pass/fail status and cleaned arrays.
    """
    result = QualityFilterResult()

    try:
        # Step 1: Quality flag masking
        # Mask all cadences where quality flag is non-zero
        # This handles: momentum dumps, cosmic ray events, scattered light,
        # and manual exclude flags
        if hasattr(lc, 'quality') and lc.quality is not None:
            quality_mask = lc.quality == 0
            lc_clean = lc[quality_mask]
        else:
            # If no quality column, use all data
            lc_clean = lc
            logger.debug("No quality column found, using all data points")

        # Remove NaN flux values
        finite_mask = np.isfinite(lc_clean.flux.value)
        if hasattr(lc_clean, 'flux_err') and lc_clean.flux_err is not None:
            finite_mask &= np.isfinite(lc_clean.flux_err.value)
        lc_clean = lc_clean[finite_mask]

        # Step 2: Minimum datapoints check
        n_points = len(lc_clean)
        result.n_valid_points = n_points

        if n_points < min_datapoints:
            result.passed = False
            result.reason = (
                f"Insufficient data points: {n_points} < {min_datapoints} "
                f"after quality masking"
            )
            logger.debug(result.reason)
            return result

        # Step 3: CROWDSAP check
        crowdsap = lc.meta.get("CROWDSAP", np.nan)
        result.crowdsap = float(crowdsap) if crowdsap is not None else np.nan

        if not np.isnan(result.crowdsap) and result.crowdsap < min_crowdsap:
            result.passed = False
            result.reason = (
                f"Contaminated pixel: CROWDSAP={result.crowdsap:.3f} < {min_crowdsap}"
            )
            logger.debug(result.reason)
            return result

        # Normalize flux by median for RMS computation
        flux_values = lc_clean.flux.value.copy()
        median_flux = np.nanmedian(flux_values)
        if median_flux > 0:
            norm_flux = flux_values / median_flux
        else:
            result.passed = False
            result.reason = "Non-positive median flux"
            return result

        # Step 4: Stellar activity check (soft flag, does NOT reject)
        raw_rms = np.nanstd(norm_flux)
        result.raw_rms = float(raw_rms)

        if raw_rms > max_rms_threshold:
            result.stellar_activity_flag = True
            logger.debug(
                f"Stellar activity flag: RMS={raw_rms:.4f} > {max_rms_threshold}"
            )

        # Store cleaned arrays
        result.time = lc_clean.time.btjd  # Barycentric TESS Julian Date
        result.flux = flux_values
        result.flux_err = (
            lc_clean.flux_err.value
            if hasattr(lc_clean, 'flux_err') and lc_clean.flux_err is not None
            else np.ones_like(flux_values) * raw_rms
        )

        result.passed = True
        logger.debug(
            f"Quality filter passed: {n_points} points, "
            f"CROWDSAP={result.crowdsap:.3f}, RMS={raw_rms:.5f}, "
            f"activity_flag={result.stellar_activity_flag}"
        )

    except Exception as e:
        result.passed = False
        result.reason = f"Quality filter error: {type(e).__name__}: {e}"
        logger.error(result.reason, exc_info=True)

    return result


def filter_from_arrays(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray | None = None,
    crowdsap: float = np.nan,
    min_datapoints: int = 100,
    min_crowdsap: float = 0.5,
    max_rms_threshold: float = 0.05,
) -> QualityFilterResult:
    """Apply quality filtering to raw arrays (alternative to lightkurve object).

    Args:
        time: Time array.
        flux: Raw flux array.
        flux_err: Flux uncertainty array.
        quality: Quality flag array (optional).
        crowdsap: CROWDSAP value.
        min_datapoints: Minimum valid data points required.
        min_crowdsap: Minimum CROWDSAP value.
        max_rms_threshold: RMS threshold for stellar activity flag.

    Returns:
        QualityFilterResult with filtered arrays.
    """
    result = QualityFilterResult()
    result.crowdsap = crowdsap

    try:
        # Quality mask
        if quality is not None:
            mask = quality == 0
        else:
            mask = np.ones(len(time), dtype=bool)

        # NaN mask
        mask &= np.isfinite(flux)
        mask &= np.isfinite(time)
        if flux_err is not None:
            mask &= np.isfinite(flux_err)

        t_clean = time[mask]
        f_clean = flux[mask]
        fe_clean = flux_err[mask] if flux_err is not None else np.ones_like(f_clean)

        result.n_valid_points = len(t_clean)

        if result.n_valid_points < min_datapoints:
            result.passed = False
            result.reason = (
                f"Insufficient data: {result.n_valid_points} < {min_datapoints}"
            )
            return result

        if not np.isnan(crowdsap) and crowdsap < min_crowdsap:
            result.passed = False
            result.reason = f"CROWDSAP={crowdsap:.3f} < {min_crowdsap}"
            return result

        # Normalize and compute RMS
        median_flux = np.nanmedian(f_clean)
        if median_flux <= 0:
            result.passed = False
            result.reason = "Non-positive median flux"
            return result

        norm_flux = f_clean / median_flux
        raw_rms = np.nanstd(norm_flux)
        result.raw_rms = float(raw_rms)
        result.stellar_activity_flag = raw_rms > max_rms_threshold

        result.time = t_clean
        result.flux = f_clean
        result.flux_err = fe_clean
        result.passed = True

    except Exception as e:
        result.passed = False
        result.reason = f"Filter error: {e}"

    return result
