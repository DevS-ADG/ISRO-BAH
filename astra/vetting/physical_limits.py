"""
ASTRA Vetting — Physical Parameter Limits (Test 06).

Enforces hard physical limits on implied planet radius and equilibrium
temperature. Objects too large or too hot are not planets.

HARD REJECTION:
  R_planet > 25 R_Earth → STELLAR_COMPANION
  T_eq > 4000 K → PHYSICALLY_EVAPORATED
"""

import numpy as np

from astra.utils.logger import get_logger
from astra.utils.stellar_utils import (
    equilibrium_temperature,
    estimate_stellar_mass,
    planet_radius_earth,
    semi_major_axis_au,
)

logger = get_logger("astra.vetting.physical_limits")


def test_physical_limits(
    depth: float,
    period: float,
    r_star: float,
    teff: float,
    r_planet_max: float = 25.0,
    t_eq_max: float = 4000.0,
) -> tuple[bool, str, float, float, bool]:
    """Check implied physical parameters against hard limits.

    Physical justification:
    - R_planet > 25 R_Earth (~2.3 R_Jupiter): Objects this large are stellar
      companions or brown dwarfs, not planets. The threshold provides margin
      for grazing geometries.
    - T_eq > 4000 K: Planets at these temperatures are photoevaporated and
      unlikely to retain atmosphere. Transit signals may be contaminated
      by stellar variability at close orbital distances.

    Args:
        depth: Fractional transit depth.
        period: Orbital period in days.
        r_star: Stellar radius in solar radii.
        teff: Stellar effective temperature in Kelvin.
        r_planet_max: Maximum planet radius in Earth radii (default 25.0).
        t_eq_max: Maximum equilibrium temperature in Kelvin (default 4000.0).

    Returns:
        Tuple of (flag_fired, rejection_cause, r_planet_earth, t_eq, hard_reject).
    """
    flag_fired = False
    rejection_cause = "NONE"
    hard_reject = False
    r_planet_val = np.nan
    t_eq_val = np.nan

    # ── Planet radius check ─────────────────────────────────────────────
    r_planet_val = planet_radius_earth(depth, r_star)

    if np.isfinite(r_planet_val) and r_planet_val > r_planet_max:
        flag_fired = True
        hard_reject = True
        rejection_cause = "STELLAR_COMPANION"
        logger.info(
            f"Physical limits FAILED: R_planet={r_planet_val:.1f} R_Earth > "
            f"{r_planet_max} -> STELLAR_COMPANION"
        )

    # ── Equilibrium temperature check ───────────────────────────────────
    if not hard_reject:
        m_star = estimate_stellar_mass(teff, r_star)
        a_au = semi_major_axis_au(period, m_star)

        t_eq_val = equilibrium_temperature(teff, r_star, a_au)

        if np.isfinite(t_eq_val) and t_eq_val > t_eq_max:
            flag_fired = True
            hard_reject = True
            rejection_cause = "PHYSICALLY_EVAPORATED"
            logger.info(
                f"Physical limits FAILED: T_eq={t_eq_val:.0f} K > "
                f"{t_eq_max} -> PHYSICALLY_EVAPORATED"
            )

    if not flag_fired:
        logger.debug(
            f"Physical limits passed: R_planet={r_planet_val:.1f} R_Earth, "
            f"T_eq={t_eq_val:.0f} K"
        )

    return flag_fired, rejection_cause, float(r_planet_val), float(t_eq_val), hard_reject
