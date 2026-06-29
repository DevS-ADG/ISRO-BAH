"""
ASTRA Vetting — Duration Consistency Check (Test 05).

Compares observed transit duration to the expected duration from Kepler's
third law and stellar density. Extreme ratios indicate physically
inconsistent geometries.

SOFT FLAG (not a hard rejection): flags for manual review.
"""

import numpy as np

from astra.utils.logger import get_logger
from astra.utils.stellar_utils import (
    estimate_stellar_mass,
    expected_transit_duration_hours,
    semi_major_axis_stellar_radii,
)

logger = get_logger("astra.vetting.duration_check")


def test_duration_consistency(
    duration_hours: float,
    period: float,
    r_star: float,
    teff: float,
    duration_ratio_min: float = 0.3,
    duration_ratio_max: float = 3.0,
) -> tuple[bool, float]:
    """Check transit duration against Kepler's third law expectation.

    Computes the expected transit duration for a central (b=0) transit and
    checks whether the observed duration is physically consistent.

    Physical justification:
    - Extremely long durations → grazing geometry likely EB.
    - Extremely short durations → implausible transit geometry.

    Args:
        duration_hours: Observed transit duration in hours.
        period: Orbital period in days.
        r_star: Stellar radius in solar radii.
        teff: Stellar effective temperature in Kelvin.
        duration_ratio_min: Minimum acceptable ratio (default 0.3).
        duration_ratio_max: Maximum acceptable ratio (default 3.0).

    Returns:
        Tuple of (flag_fired, duration_ratio).
        flag_fired: True if ratio is outside [min, max] bounds.
    """
    if (
        np.isnan(duration_hours)
        or np.isnan(period)
        or np.isnan(r_star)
        or np.isnan(teff)
        or duration_hours <= 0
        or period <= 0
        or r_star <= 0
    ):
        return False, np.nan

    # Estimate stellar mass from Teff and R_star
    m_star = estimate_stellar_mass(teff, r_star)

    # Compute expected duration for central transit (b=0)
    a_rs = semi_major_axis_stellar_radii(period, m_star, r_star)

    if np.isnan(a_rs) or a_rs <= 0:
        return False, np.nan

    t_expected = expected_transit_duration_hours(period, r_star, a_rs, b=0.0)

    if np.isnan(t_expected) or t_expected <= 0:
        return False, np.nan

    # Duration ratio
    duration_ratio = duration_hours / t_expected

    # Flag if outside bounds
    flag_fired = duration_ratio < duration_ratio_min or duration_ratio > duration_ratio_max

    if flag_fired:
        logger.info(
            f"Duration check soft flag: ratio={duration_ratio:.2f} "
            f"(observed={duration_hours:.2f}h, expected={t_expected:.2f}h) "
            f"outside [{duration_ratio_min}, {duration_ratio_max}]"
        )
    else:
        logger.debug(
            f"Duration check passed: ratio={duration_ratio:.2f} "
            f"(observed={duration_hours:.2f}h, expected={t_expected:.2f}h)"
        )

    return flag_fired, float(duration_ratio)
