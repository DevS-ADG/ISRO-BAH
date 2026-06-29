"""
ASTRA Stellar Utilities — Physical constants, stellar parameter estimation,
unit conversions, and Kepler's third law utilities.

All constants use SI-compatible units with clear documentation.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.utils.stellar")

# ──────────────────────────────────────────────────────────────────────────────
# Physical Constants
# ──────────────────────────────────────────────────────────────────────────────

R_SUN_METERS = 6.957e8          # Solar radius in meters
R_SUN_EARTH = 109.076           # Solar radius in Earth radii
R_EARTH_METERS = 6.371e6        # Earth radius in meters
M_SUN_KG = 1.989e30             # Solar mass in kg
AU_METERS = 1.496e11            # Astronomical unit in meters
G_SI = 6.674e-11                # Gravitational constant in m^3 kg^-1 s^-2
SIGMA_SB = 5.670e-8             # Stefan-Boltzmann constant in W m^-2 K^-4
DAY_SECONDS = 86400.0           # Seconds per day
TESS_PLATE_SCALE = 21.0         # TESS plate scale in arcsec/pixel

# Default assumptions (documented as per spec)
# Assumption: geometric albedo 0.3 is a standard first-order approximation
DEFAULT_ALBEDO = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Stellar Parameter Estimation
# ──────────────────────────────────────────────────────────────────────────────


def estimate_stellar_mass(teff: float, r_star: float) -> float:
    """Estimate stellar mass from effective temperature and radius.

    Uses empirical main-sequence mass-luminosity-temperature relations.
    This is a first-order approximation for solar-type stars.

    For main-sequence stars, the mass-radius relation approximately follows:
        M ∝ R^1.25 for R < 1 R_sun
        M ∝ R^0.8  for R >= 1 R_sun

    Combined with a temperature correction factor.

    Args:
        teff: Effective temperature in Kelvin.
        r_star: Stellar radius in solar radii.

    Returns:
        Estimated stellar mass in solar masses.
    """
    if np.isnan(teff) or np.isnan(r_star) or teff <= 0 or r_star <= 0:
        logger.warning(
            f"Invalid stellar parameters (Teff={teff}, R_star={r_star}), "
            "defaulting to solar mass"
        )
        return 1.0

    # Temperature-based correction relative to solar (5778 K)
    temp_ratio = teff / 5778.0

    if r_star < 1.0:
        mass = r_star ** 1.25 * temp_ratio ** 0.5
    else:
        mass = r_star ** 0.8 * temp_ratio ** 0.5

    # Clamp to physically reasonable range
    mass = np.clip(mass, 0.08, 100.0)

    return float(mass)


def semi_major_axis_au(period_days: float, m_star_solar: float) -> float:
    """Compute semi-major axis from Kepler's third law.

    a³ = (G × M_star × P²) / (4π²)

    Args:
        period_days: Orbital period in days.
        m_star_solar: Stellar mass in solar masses.

    Returns:
        Semi-major axis in AU.
    """
    if period_days <= 0 or m_star_solar <= 0:
        return np.nan

    period_s = period_days * DAY_SECONDS
    m_star_kg = m_star_solar * M_SUN_KG

    a_meters = (G_SI * m_star_kg * period_s ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    return a_meters / AU_METERS


def semi_major_axis_stellar_radii(
    period_days: float, m_star_solar: float, r_star_solar: float
) -> float:
    """Compute semi-major axis in units of stellar radii.

    Args:
        period_days: Orbital period in days.
        m_star_solar: Stellar mass in solar masses.
        r_star_solar: Stellar radius in solar radii.

    Returns:
        Semi-major axis in stellar radii (a/R_star).
    """
    a_au = semi_major_axis_au(period_days, m_star_solar)
    if np.isnan(a_au) or r_star_solar <= 0:
        return np.nan

    a_meters = a_au * AU_METERS
    r_star_meters = r_star_solar * R_SUN_METERS

    return a_meters / r_star_meters


def planet_radius_earth(depth: float, r_star_solar: float) -> float:
    """Compute implied planet radius from transit depth.

    R_planet = R_star × √(depth) × (R_sun / R_earth)

    Args:
        depth: Fractional transit depth (e.g., 0.01 for 1% dip).
        r_star_solar: Stellar radius in solar radii.

    Returns:
        Planet radius in Earth radii.
    """
    if depth <= 0 or r_star_solar <= 0 or np.isnan(depth) or np.isnan(r_star_solar):
        return np.nan

    # Rp/Rs = sqrt(depth), then convert to Earth radii
    rp_rs = np.sqrt(depth)
    r_planet_solar = rp_rs * r_star_solar
    return r_planet_solar * R_SUN_EARTH


def equilibrium_temperature(
    teff: float,
    r_star_solar: float,
    a_au: float,
    albedo: float = DEFAULT_ALBEDO,
) -> float:
    """Compute planetary equilibrium temperature.

    T_eq = T_eff × √(R_star / (2 × a)) × (1 - albedo)^0.25

    Assumption: geometric albedo 0.3 is a standard first-order approximation.

    Args:
        teff: Stellar effective temperature in Kelvin.
        r_star_solar: Stellar radius in solar radii.
        a_au: Semi-major axis in AU.
        albedo: Planetary geometric albedo (default 0.3).

    Returns:
        Equilibrium temperature in Kelvin.
    """
    if (
        teff <= 0
        or r_star_solar <= 0
        or a_au <= 0
        or np.isnan(teff)
        or np.isnan(r_star_solar)
        or np.isnan(a_au)
    ):
        return np.nan

    r_star_meters = r_star_solar * R_SUN_METERS
    a_meters = a_au * AU_METERS

    t_eq = teff * np.sqrt(r_star_meters / (2.0 * a_meters)) * (1.0 - albedo) ** 0.25

    return float(t_eq)


def expected_transit_duration_hours(
    period_days: float,
    r_star_solar: float,
    a_rs: float,
    b: float = 0.0,
    rp_rs: float = 0.01,
) -> float:
    """Compute expected transit duration for given orbital parameters.

    T_dur = (P / π) × arcsin((R_star / a) × √((1 + Rp/Rs)² - b²))

    Simplified for small Rp/Rs and b=0:
        T_dur ≈ (R_star × P) / (π × a)

    Args:
        period_days: Orbital period in days.
        r_star_solar: Stellar radius in solar radii.
        a_rs: Semi-major axis in stellar radii (a/R_star).
        b: Impact parameter (0 = central transit).
        rp_rs: Planet-to-star radius ratio.

    Returns:
        Expected transit duration in hours.
    """
    if period_days <= 0 or a_rs <= 0 or np.isnan(period_days) or np.isnan(a_rs):
        return np.nan

    # Full formula: T = (P/π) × arcsin((1/a_rs) × sqrt((1+rp_rs)² - b²))
    arg = (1.0 / a_rs) * np.sqrt(max(0, (1.0 + rp_rs) ** 2 - b ** 2))

    # Clamp argument to valid arcsin range
    arg = np.clip(arg, -1.0, 1.0)

    t_dur_days = (period_days / np.pi) * np.arcsin(arg)
    return t_dur_days * 24.0  # Convert to hours


def compute_snr(
    depth: float, n_in_transit: int, rms_out_of_transit: float
) -> float:
    """Compute transit signal-to-noise ratio.

    SNR = depth × √(N_in_transit) / RMS_out_of_transit

    Args:
        depth: Fractional transit depth.
        n_in_transit: Total number of in-transit data points across all transits.
        rms_out_of_transit: Standard deviation of detrended out-of-transit flux.

    Returns:
        Signal-to-noise ratio.
    """
    if (
        depth <= 0
        or n_in_transit <= 0
        or rms_out_of_transit <= 0
        or np.isnan(depth)
        or np.isnan(rms_out_of_transit)
    ):
        return 0.0

    return depth * np.sqrt(n_in_transit) / rms_out_of_transit


def impact_parameter_proxy(
    duration_hours: float,
    period_days: float,
    r_star_solar: float,
    m_star_solar: float,
) -> float:
    """Estimate impact parameter from transit duration ratio.

    The proxy is derived from the ratio of observed duration to the
    expected duration for a central (b=0) transit.

    b_proxy = √(1 - (T_obs / T_central)²)

    Args:
        duration_hours: Observed transit duration in hours.
        period_days: Orbital period in days.
        r_star_solar: Stellar radius in solar radii.
        m_star_solar: Stellar mass in solar masses.

    Returns:
        Estimated impact parameter (0 to ~1).
    """
    if duration_hours <= 0 or period_days <= 0:
        return np.nan

    a_rs = semi_major_axis_stellar_radii(period_days, m_star_solar, r_star_solar)
    if np.isnan(a_rs) or a_rs <= 0:
        return np.nan

    t_central = expected_transit_duration_hours(period_days, r_star_solar, a_rs, b=0.0)

    if np.isnan(t_central) or t_central <= 0:
        return np.nan

    ratio = duration_hours / t_central

    # b = sqrt(1 - ratio^2), clamped to [0, 1]
    if ratio >= 1.0:
        return 0.0

    b_proxy = np.sqrt(max(0, 1.0 - ratio ** 2))
    return float(np.clip(b_proxy, 0.0, 1.5))
