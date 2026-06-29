"""
ASTRA Vetting — Odd-Even Depth Test (Test 01).

Detects eclipsing binaries by comparing transit depths at odd vs even
numbered transits. In an EB at half the true period, primary and secondary
eclipses alternate as deep and shallow transits. A true planet produces
identical depth transits to measurement precision.

HARD REJECTION: sigma > threshold → ECLIPSING_BINARY
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.vetting.odd_even")


def test_odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    transit_times: np.ndarray,
    period: float,
    duration: float,
    sigma_threshold: float = 3.0,
) -> tuple[bool, float, bool]:
    """Run the odd-even transit depth test.

    Splits transit events into odd (1st, 3rd, 5th...) and even (2nd, 4th, 6th...)
    groups, measures mean depth in each, and computes statistical significance
    of the difference.

    Physical justification: In an eclipsing binary where two unequal stars
    eclipse each other, the primary and secondary eclipses have different depths.
    At half the true period, this appears as alternating deep/shallow transits.

    Args:
        time: Full time array.
        flux: Full flux array (detrended, normalized).
        transit_times: Array of individual transit mid-times.
        period: Orbital period in days.
        duration: Transit duration in days.
        sigma_threshold: Significance threshold for hard rejection (default 3.0).

    Returns:
        Tuple of (flag_fired, sigma_value, hard_reject).
        flag_fired: True if the test detected a significant difference.
        sigma_value: Statistical significance of the depth difference.
        hard_reject: True if sigma > threshold (ECLIPSING_BINARY).
    """
    if len(transit_times) < 4:
        # Need at least 2 odd and 2 even transits
        logger.debug("Odd-even test: fewer than 4 transits, skipping")
        return False, np.nan, False

    # Split into odd and even transit indices (1-indexed: 1st=odd, 2nd=even, etc.)
    odd_times = transit_times[0::2]   # 1st, 3rd, 5th...
    even_times = transit_times[1::2]  # 2nd, 4th, 6th...

    half_dur = duration / 2.0

    # Compute mean depth for odd transits
    odd_depths = []
    for t_mid in odd_times:
        mask = np.abs(time - t_mid) <= half_dur
        if np.sum(mask) > 2:
            in_transit_flux = flux[mask]
            # Depth = 1.0 - mean(in_transit_flux) for normalized flux
            depth = 1.0 - np.nanmean(in_transit_flux)
            odd_depths.append(depth)

    # Compute mean depth for even transits
    even_depths = []
    for t_mid in even_times:
        mask = np.abs(time - t_mid) <= half_dur
        if np.sum(mask) > 2:
            in_transit_flux = flux[mask]
            depth = 1.0 - np.nanmean(in_transit_flux)
            even_depths.append(depth)

    if len(odd_depths) < 1 or len(even_depths) < 1:
        logger.debug("Odd-even test: insufficient data in one group")
        return False, np.nan, False

    # Compute mean and standard error for each group
    depth_odd = np.mean(odd_depths)
    depth_even = np.mean(even_depths)
    sigma_odd = np.std(odd_depths) / np.sqrt(len(odd_depths)) if len(odd_depths) > 1 else np.std(odd_depths)
    sigma_even = np.std(even_depths) / np.sqrt(len(even_depths)) if len(even_depths) > 1 else np.std(even_depths)

    # Significance of depth difference
    denominator = np.sqrt(sigma_odd**2 + sigma_even**2)
    if denominator <= 0 or not np.isfinite(denominator):
        return False, np.nan, False

    sigma_value = abs(depth_odd - depth_even) / denominator

    flag_fired = sigma_value > sigma_threshold
    hard_reject = flag_fired  # This is a hard rejection test

    if hard_reject:
        logger.info(
            f"Odd-even test FAILED: σ={sigma_value:.2f} > {sigma_threshold} "
            f"(depth_odd={depth_odd:.5f}, depth_even={depth_even:.5f}) → ECLIPSING_BINARY"
        )
    else:
        logger.debug(
            f"Odd-even test passed: σ={sigma_value:.2f} ≤ {sigma_threshold}"
        )

    return flag_fired, float(sigma_value), hard_reject
