"""
ASTRA Detrending — Systematic noise removal from TESS light curves.

Implements the 4-step detrending process:
1. Sigma clipping (5σ) to remove residual outliers
2. Gap handling (segment at gaps > 0.5 days)
3. Wotan biweight detrending (0.75-day window)
4. Normalization to baseline 1.0

The biweight window (0.75 days) is chosen to be significantly wider
than the maximum expected transit duration (~2-4 hours) so the filter
does not absorb transit dips.
"""

import numpy as np
from wotan import flatten

from astra.preprocessing.gap_handler import (
    reassemble_segments,
    segment_light_curve,
)
from astra.preprocessing.normalization import normalize_flux, sigma_clip
from astra.utils.logger import get_logger

logger = get_logger("astra.preprocessing.detrending")


class DetrendingResult:
    """Result of the detrending pipeline for a single star.

    Attributes:
        time: Cleaned time array (BTJD).
        flux: Detrended, normalized flux array (baseline ~1.0).
        flux_err: Scaled flux uncertainty array.
        trend: Estimated systematic trend array.
        normalization_factor: Factor used for normalization.
        n_clipped: Number of points removed by sigma clipping.
        n_segments: Number of continuous segments after gap handling.
    """

    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.flux: np.ndarray = np.array([])
        self.flux_err: np.ndarray = np.array([])
        self.trend: np.ndarray = np.array([])
        self.normalization_factor: float = 1.0
        self.n_clipped: int = 0
        self.n_segments: int = 0


def detrend_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sigma_clip_threshold: float = 5.0,
    detrend_method: str = "biweight",
    detrend_window_days: float = 0.75,
    gap_threshold_days: float = 0.5,
) -> DetrendingResult:
    """Apply the full 4-step detrending pipeline to a light curve.

    Args:
        time: Time array in days (BTJD).
        flux: Raw flux array.
        flux_err: Flux uncertainty array.
        sigma_clip_threshold: Sigma clipping threshold (default 5.0).
        detrend_method: Detrending method ('biweight' or 'spline').
        detrend_window_days: Window length for the detrending filter in days.
        gap_threshold_days: Minimum gap to trigger segmentation.

    Returns:
        DetrendingResult with detrended, normalized arrays.
    """
    result = DetrendingResult()

    # ── Step 1: Sigma Clipping ──────────────────────────────────────────
    # Remove flux values beyond sigma_clip_threshold from the running median.
    # This eliminates residual cosmic ray hits and flare spikes not caught
    # by TESS quality flags.
    clip_mask = sigma_clip(flux, sigma=sigma_clip_threshold)
    result.n_clipped = int(np.sum(~clip_mask))

    time_clipped = time[clip_mask]
    flux_clipped = flux[clip_mask]
    flux_err_clipped = flux_err[clip_mask]

    if len(time_clipped) < 50:
        logger.warning(
            f"Too few points after sigma clipping: {len(time_clipped)}"
        )
        result.time = time_clipped
        result.flux = flux_clipped / np.nanmedian(flux_clipped)
        result.flux_err = flux_err_clipped / np.nanmedian(flux_clipped)
        result.trend = np.ones_like(flux_clipped)
        return result

    # ── Step 2: Gap Handling ────────────────────────────────────────────
    # Identify time gaps > gap_threshold_days and treat each continuous
    # segment independently during detrending to prevent filter edge
    # effects from propagating across gaps.
    segments = segment_light_curve(
        time_clipped,
        flux_clipped,
        flux_err_clipped,
        gap_threshold_days=gap_threshold_days,
        min_segment_points=20,
    )
    result.n_segments = len(segments)

    if len(segments) == 0:
        logger.warning("No valid segments found after gap handling")
        result.time = time_clipped
        result.flux = flux_clipped / np.nanmedian(flux_clipped)
        result.flux_err = flux_err_clipped / np.nanmedian(flux_clipped)
        result.trend = np.ones_like(flux_clipped)
        return result

    # ── Step 3: Detrending (per segment) ────────────────────────────────
    # Apply wotan's flatten() with biweight method and the configured
    # window length. The biweight location estimator is robust to outliers.
    # The window is wider than maximum transit duration (~2-4 hours = ~0.08-0.17 days)
    # so the filter does not absorb transit dips.
    detrended_segments: list[dict] = []

    for i, seg in enumerate(segments):
        seg_time = seg["time"]
        seg_flux = seg["flux"]
        seg_flux_err = seg["flux_err"]

        try:
            flat_flux, trend_flux = flatten(
                seg_time,
                seg_flux,
                method=detrend_method,
                window_length=detrend_window_days,
                return_trend=True,
            )

            # Handle any NaN values from the detrending
            valid = np.isfinite(flat_flux) & np.isfinite(trend_flux)

            if np.sum(valid) < 10:
                logger.debug(
                    f"Segment {i}: too few valid points after detrending, "
                    "using raw flux"
                )
                trend_flux = np.ones_like(seg_flux) * np.nanmedian(seg_flux)
                flat_flux = seg_flux / trend_flux
                valid = np.isfinite(flat_flux)

            detrended_segments.append(
                {
                    "time": seg_time[valid],
                    "flux": flat_flux[valid],
                    "flux_err": seg_flux_err[valid] / trend_flux[valid],
                    "trend": trend_flux[valid],
                }
            )

        except Exception as e:
            logger.warning(
                f"Detrending failed for segment {i}: {e}. Using raw flux."
            )
            # Fallback: simple median normalization
            median = np.nanmedian(seg_flux)
            if median > 0:
                detrended_segments.append(
                    {
                        "time": seg_time,
                        "flux": seg_flux / median,
                        "flux_err": seg_flux_err / median,
                        "trend": np.ones_like(seg_flux) * median,
                    }
                )

    # Reassemble all segments
    if not detrended_segments:
        logger.error("All detrending segments failed")
        result.time = time_clipped
        result.flux = flux_clipped / np.nanmedian(flux_clipped)
        result.flux_err = flux_err_clipped / np.nanmedian(flux_clipped)
        result.trend = np.ones_like(flux_clipped)
        return result

    reassembled = reassemble_segments(
        detrended_segments, keys=["time", "flux", "flux_err", "trend"]
    )

    # ── Step 4: Normalization ───────────────────────────────────────────
    # Normalize the detrended flux so the out-of-transit baseline is 1.0.
    norm_flux, norm_err, norm_factor = normalize_flux(
        reassembled["flux"],
        reassembled["flux_err"],
        method="median",
    )

    result.time = reassembled["time"]
    result.flux = norm_flux
    result.flux_err = norm_err if norm_err is not None else reassembled["flux_err"]
    result.trend = reassembled["trend"]
    result.normalization_factor = norm_factor

    logger.debug(
        f"Detrending complete: {len(result.time)} points, "
        f"{result.n_segments} segments, {result.n_clipped} clipped, "
        f"method={detrend_method}, window={detrend_window_days}d"
    )

    return result
