"""
ASTRA Vetting — Secondary Eclipse Check (Test 02).

Detects eclipsing binaries by measuring flux at phase 0.5 (halfway between
primary transits). A planet produces no secondary eclipse; an EB produces
one when the secondary star passes behind the primary.

HARD REJECTION: sigma > threshold → ECLIPSING_BINARY
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.vetting.secondary_eclipse")


def test_secondary_eclipse(
    phase: np.ndarray,
    flux: np.ndarray,
    primary_depth: float,
    duration_phase: float,
    sigma_threshold: float = 3.0,
) -> tuple[bool, float, float, bool]:
    """Check for a secondary eclipse at phase 0.5.

    Measures the flux dip at orbital phase 0.5 and computes its statistical
    significance relative to the out-of-transit noise.

    Physical justification: A planet is dark relative to the star, so no
    secondary eclipse is expected. An eclipsing binary produces a secondary
    eclipse when the dimmer secondary star passes behind the primary.

    Args:
        phase: Phase-folded phase array (−0.5 to 0.5).
        flux: Phase-folded flux array.
        primary_depth: Primary transit depth (fractional).
        duration_phase: Transit duration in phase units (duration / period).
        sigma_threshold: Significance threshold for hard rejection (default 3.0).

    Returns:
        Tuple of (flag_fired, sigma_secondary, secondary_depth_ratio, hard_reject).
    """
    if len(phase) < 20 or np.isnan(primary_depth) or primary_depth <= 0:
        return False, np.nan, np.nan, False

    half_dur = duration_phase / 2.0

    # ── Measure flux at phase 0.5 (anti-transit position) ───────────────
    # For phase centered at 0 with range [-0.5, 0.5], the anti-transit
    # is near ±0.5. Check both edges:
    secondary_mask = (np.abs(phase) >= (0.5 - half_dur))
    oot_mask = (np.abs(phase) > half_dur * 1.5) & (np.abs(phase) < (0.5 - half_dur * 1.5))

    if np.sum(secondary_mask) < 3 or np.sum(oot_mask) < 10:
        logger.debug("Secondary eclipse test: insufficient data points")
        return False, np.nan, np.nan, False

    flux_at_secondary = flux[secondary_mask]
    flux_oot = flux[oot_mask]

    # Out-of-transit baseline
    median_oot = np.nanmedian(flux_oot)
    rms_oot = np.nanstd(flux_oot)

    # Secondary eclipse depth
    mean_secondary = np.nanmean(flux_at_secondary)
    n_secondary = np.sum(secondary_mask)

    noise_floor = rms_oot / np.sqrt(max(1, n_secondary))

    if noise_floor <= 0:
        return False, np.nan, np.nan, False

    # Significance of flux dip at phase 0.5
    sigma_secondary = (median_oot - mean_secondary) / noise_floor

    # Secondary depth ratio
    secondary_depth = max(0, median_oot - mean_secondary)
    secondary_depth_ratio = secondary_depth / primary_depth if primary_depth > 0 else 0.0

    flag_fired = sigma_secondary > sigma_threshold
    hard_reject = flag_fired

    if hard_reject:
        logger.info(
            f"Secondary eclipse test FAILED: sigma={sigma_secondary:.2f} > "
            f"{sigma_threshold}, depth_ratio={secondary_depth_ratio:.3f} -> ECLIPSING_BINARY"
        )
    else:
        logger.debug(
    return flag_fired, float(sigma_secondary), float(secondary_depth_ratio), hard_reject
