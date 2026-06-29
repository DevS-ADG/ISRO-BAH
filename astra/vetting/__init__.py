"""
ASTRA Vetting Module — Astrophysical false-positive rejection system.

Runs all 6 vetting tests sequentially and produces a summary result.
Hard-rejected candidates bypass ML classification and are written
directly to the catalogue with the rejection cause.

Tests:
  01. Odd-Even Depth (HARD) — Eclipsing binary detection
  02. Secondary Eclipse (HARD) — EB at phase 0.5
  03. Centroid Shift (HARD) — Background eclipsing binary
  04. Shape Analysis (SOFT) — V-shape transit flag
  05. Duration Check (SOFT) — Physical consistency
  06. Physical Limits (HARD) — R_planet and T_eq bounds
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from astra.vetting.odd_even import test_odd_even
from astra.vetting.secondary_eclipse import test_secondary_eclipse
from astra.vetting.centroid import test_centroid_shift
from astra.vetting.shape_analysis import test_shape_analysis
from astra.vetting.duration_check import test_duration_consistency
from astra.vetting.physical_limits import test_physical_limits
from astra.utils.logger import get_logger

logger = get_logger("astra.vetting")


@dataclass
class VettingResult:
    """Summary of all astrophysical vetting tests for a candidate.

    Attributes:
        hard_rejected: True if any hard rejection test fired.
        rejection_cause: Name of the first test that triggered hard rejection.
        soft_flags: List of test names that returned soft flags.
        vetting_passed: True if hard_rejected is False.
        test_results: Dictionary of individual test results.
        odd_even_sigma: Odd-even depth difference significance.
        secondary_depth_ratio: Secondary eclipse depth / primary depth.
        centroid_shift: Centroid shift in arcseconds.
        flat_bottom_ratio: Flat bottom width / total transit width.
        ingress_symmetry: Ingress-egress Pearson correlation.
        duration_ratio: Observed / expected transit duration.
        r_planet_earth: Implied planet radius in Earth radii.
        t_eq: Equilibrium temperature in Kelvin.
    """

    hard_rejected: bool = False
    rejection_cause: str = "NONE"
    soft_flags: list[str] = field(default_factory=list)
    vetting_passed: bool = True
    test_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Feature values populated by vetting tests
    odd_even_sigma: float = np.nan
    secondary_depth_ratio: float = np.nan
    centroid_shift: float = np.nan
    flat_bottom_ratio: float = np.nan
    ingress_symmetry: float = np.nan
    duration_ratio: float = np.nan
    r_planet_earth: float = np.nan
    t_eq: float = np.nan


def run_all_vetting(
    time: np.ndarray,
    flux: np.ndarray,
    phase: np.ndarray,
    phase_flux: np.ndarray,
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    transit_times: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    depth: float,
    r_star: float,
    teff: float,
    tpf: Any = None,
    config: dict | None = None,
) -> VettingResult:
    """Run all 6 astrophysical vetting tests on a candidate.

    Tests run sequentially. Each test produces a (flag, value, pass/fail) tuple.
    Hard rejection occurs on the first failing hard test. All tests still run
    to populate the feature vector even after a hard rejection.

    Args:
        time: Full detrended time array.
        flux: Full detrended flux array.
        phase: Phase-folded phase array.
        phase_flux: Phase-folded flux array.
        binned_phase: Binned phase-folded phase array.
        binned_flux: Binned phase-folded flux array.
        transit_times: Individual transit mid-times.
        period: Orbital period in days.
        t0: Transit mid-time.
        duration_days: Transit duration in days.
        depth: Transit depth (fractional).
        r_star: Stellar radius in solar radii.
        teff: Effective temperature in Kelvin.
        tpf: Target Pixel File (lightkurve TPF object or None).
        config: Vetting configuration dictionary (thresholds).

    Returns:
        VettingResult with all test outcomes and feature values.
    """
    if config is None:
        config = {}

    result = VettingResult()

    # ── Test 01: Odd-Even Depth ─────────────────────────────────────────
    try:
        flag, sigma, hard = test_odd_even(
            time, flux, transit_times, period, duration_days,
            sigma_threshold=config.get("odd_even_sigma_threshold", 3.0),
        )
        result.odd_even_sigma = sigma
        result.test_results["odd_even"] = {
            "flag": flag, "value": sigma, "hard_reject": hard,
            "threshold": config.get("odd_even_sigma_threshold", 3.0),
            "status": "FAIL" if hard else "PASS",
        }
        if hard and not result.hard_rejected:
            result.hard_rejected = True
            result.rejection_cause = "ECLIPSING_BINARY_ODD_EVEN"
    except Exception as e:
        logger.error(f"Odd-even test error: {e}", exc_info=True)
        result.test_results["odd_even"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Test 02: Secondary Eclipse ──────────────────────────────────────
    try:
        duration_phase = duration_days / period if period > 0 else 0.1
        flag, sigma, depth_ratio, hard = test_secondary_eclipse(
            phase, phase_flux, depth, duration_phase,
            sigma_threshold=config.get("secondary_eclipse_sigma_threshold", 3.0),
        )
        result.secondary_depth_ratio = depth_ratio
        result.test_results["secondary_eclipse"] = {
            "flag": flag, "value": sigma, "depth_ratio": depth_ratio,
            "hard_reject": hard,
            "threshold": config.get("secondary_eclipse_sigma_threshold", 3.0),
            "status": "FAIL" if hard else "PASS",
        }
        if hard and not result.hard_rejected:
            result.hard_rejected = True
            result.rejection_cause = "ECLIPSING_BINARY_SECONDARY"
    except Exception as e:
        logger.error(f"Secondary eclipse test error: {e}", exc_info=True)
        result.test_results["secondary_eclipse"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Test 03: Centroid Shift ─────────────────────────────────────────
    try:
        flag, shift_arcsec, sigma, hard = test_centroid_shift(
            tpf, time, period, t0, duration_days,
            sigma_threshold=config.get("centroid_shift_sigma_threshold", 3.0),
        )
        result.centroid_shift = shift_arcsec
        result.test_results["centroid"] = {
            "flag": flag, "value": shift_arcsec, "sigma": sigma,
            "hard_reject": hard,
            "threshold": config.get("centroid_shift_sigma_threshold", 3.0),
            "status": "FAIL" if hard else ("SKIP" if np.isnan(shift_arcsec) else "PASS"),
        }
        if hard and not result.hard_rejected:
            result.hard_rejected = True
            result.rejection_cause = "BACKGROUND_ECLIPSING_BINARY"
    except Exception as e:
        logger.error(f"Centroid test error: {e}", exc_info=True)
        result.test_results["centroid"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Test 04: Shape Analysis (SOFT) ──────────────────────────────────
    try:
        flag, fb_ratio, sym = test_shape_analysis(
            binned_phase, binned_flux, period, duration_days,
            flat_bottom_min_ratio=config.get("shape_flat_bottom_min_ratio", 0.05),
        )
        result.flat_bottom_ratio = fb_ratio
        result.ingress_symmetry = sym
        result.test_results["shape"] = {
            "flag": flag, "flat_bottom_ratio": fb_ratio,
            "ingress_symmetry": sym, "hard_reject": False,
            "threshold": config.get("shape_flat_bottom_min_ratio", 0.05),
            "status": "FLAG" if flag else "PASS",
        }
        if flag:
            result.soft_flags.append("SHAPE_V_TRANSIT")
    except Exception as e:
        logger.error(f"Shape analysis error: {e}", exc_info=True)
        result.test_results["shape"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Test 05: Duration Consistency (SOFT) ────────────────────────────
    try:
        duration_hours = duration_days * 24.0
        flag, dur_ratio = test_duration_consistency(
            duration_hours, period, r_star, teff,
            duration_ratio_min=config.get("duration_ratio_min", 0.3),
            duration_ratio_max=config.get("duration_ratio_max", 3.0),
        )
        result.duration_ratio = dur_ratio
        result.test_results["duration"] = {
            "flag": flag, "value": dur_ratio, "hard_reject": False,
            "threshold_min": config.get("duration_ratio_min", 0.3),
            "threshold_max": config.get("duration_ratio_max", 3.0),
            "status": "FLAG" if flag else "PASS",
        }
        if flag:
            result.soft_flags.append("DURATION_INCONSISTENT")
    except Exception as e:
        logger.error(f"Duration check error: {e}", exc_info=True)
        result.test_results["duration"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Test 06: Physical Limits (HARD) ─────────────────────────────────
    try:
        flag, cause, r_pl, t_eq_val, hard = test_physical_limits(
            depth, period, r_star, teff,
            r_planet_max=config.get("r_planet_max_earth_radii", 25.0),
            t_eq_max=config.get("t_eq_max_kelvin", 4000.0),
        )
        result.r_planet_earth = r_pl
        result.t_eq = t_eq_val
        result.test_results["physical_limits"] = {
            "flag": flag, "r_planet_earth": r_pl, "t_eq": t_eq_val,
            "rejection_cause": cause, "hard_reject": hard,
            "status": "FAIL" if hard else "PASS",
        }
        if hard and not result.hard_rejected:
            result.hard_rejected = True
            result.rejection_cause = cause
    except Exception as e:
        logger.error(f"Physical limits test error: {e}", exc_info=True)
        result.test_results["physical_limits"] = {"flag": False, "value": np.nan, "status": "ERROR"}

    # ── Summary ─────────────────────────────────────────────────────────
    result.vetting_passed = not result.hard_rejected

    n_passed = sum(
        1 for t in result.test_results.values()
        if t.get("status") == "PASS"
    )
    n_failed = sum(
        1 for t in result.test_results.values()
        if t.get("status") == "FAIL"
    )
    n_flagged = sum(
        1 for t in result.test_results.values()
        if t.get("status") == "FLAG"
    )

    logger.info(
        f"Vetting summary: {'REJECTED' if result.hard_rejected else 'PASSED'} "
        f"({n_passed} passed, {n_failed} failed, {n_flagged} flagged) "
        f"{'cause=' + result.rejection_cause if result.hard_rejected else ''}"
    )

    return result
