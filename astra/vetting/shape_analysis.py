"""
ASTRA Vetting — Light Curve Shape Analysis (Test 04).

Fits a trapezoidal model to the phase-folded transit and analyzes the
shape for planetary vs EB characteristics.

SOFT FLAG (not a hard rejection): flat_bottom_ratio < 0.05 → potential EB.
"""

import numpy as np
from scipy.optimize import curve_fit

from astra.utils.logger import get_logger

logger = get_logger("astra.vetting.shape_analysis")


def test_shape_analysis(
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    period: float,
    duration_days: float,
    flat_bottom_min_ratio: float = 0.05,
) -> tuple[bool, float, float]:
    """Analyze the transit shape for planet vs EB discrimination.

    Fits a trapezoidal model and computes:
    - flat_bottom_ratio: W_flat / W_total (near 0 = V-shape EB, near 1 = flat planet)
    - ingress_symmetry: Pearson correlation between ingress and flipped egress

    Physical justification: A planetary transit has a flat bottom (planet
    fully in front of star) and symmetric ingress/egress. An eclipsing
    binary often produces a V-shaped or curved light curve.

    Args:
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        period: Orbital period in days.
        duration_days: Transit duration in days.
        flat_bottom_min_ratio: Minimum ratio to flag as potential EB.

    Returns:
        Tuple of (flag_fired, flat_bottom_ratio, ingress_symmetry).
        flag_fired: True if flat_bottom_ratio < flat_bottom_min_ratio.
    """
    if len(binned_phase) < 10 or np.isnan(period) or np.isnan(duration_days):
        return False, np.nan, np.nan

    duration_phase = duration_days / period
    half_dur = duration_phase / 2.0

    # ── Flat Bottom Ratio ───────────────────────────────────────────────
    flat_bottom_ratio = _compute_flat_bottom_ratio(
        binned_phase, binned_flux, half_dur
    )

    # ── Ingress Symmetry ────────────────────────────────────────────────
    ingress_symmetry = _compute_ingress_symmetry(
        binned_phase, binned_flux, half_dur
    )

    # ── Flag check ──────────────────────────────────────────────────────
    flag_fired = False
    if np.isfinite(flat_bottom_ratio) and flat_bottom_ratio < flat_bottom_min_ratio:
        flag_fired = True
        logger.info(
            f"Shape test soft flag: flat_bottom_ratio={flat_bottom_ratio:.3f} "
            f"< {flat_bottom_min_ratio} → potential EB (V-shape)"
        )
    else:
        logger.debug(
            f"Shape test: flat_bottom_ratio={flat_bottom_ratio:.3f}, "
            f"ingress_symmetry={ingress_symmetry:.3f}"
        )

    return flag_fired, float(flat_bottom_ratio), float(ingress_symmetry)


def _compute_flat_bottom_ratio(
    phase: np.ndarray,
    flux: np.ndarray,
    half_dur: float,
) -> float:
    """Compute the flat bottom ratio from a trapezoidal fit.

    Args:
        phase: Binned phase array.
        flux: Binned flux array.
        half_dur: Half-duration in phase units.

    Returns:
        Flat bottom ratio (0 to 1).
    """
    try:
        # Select in-transit data
        in_transit = np.abs(phase) <= half_dur
        if np.sum(in_transit) < 5:
            return np.nan

        t_phase = phase[in_transit]
        t_flux = flux[in_transit]

        # Fit a trapezoid model
        def trapezoid(x, depth, w_flat_half, t_ingress, baseline):
            """Trapezoidal transit model.

            Args:
                x: Phase values.
                depth: Transit depth.
                w_flat_half: Half-width of flat bottom in phase units.
                t_ingress: Ingress/egress duration in phase units.
                baseline: Out-of-transit baseline flux.
            """
            y = np.ones_like(x) * baseline
            abs_x = np.abs(x)

            # Flat bottom
            flat = abs_x <= w_flat_half
            y[flat] = baseline - depth

            # Ingress/egress (linear ramp)
            ramp = (abs_x > w_flat_half) & (abs_x <= w_flat_half + t_ingress)
            if t_ingress > 0:
                y[ramp] = baseline - depth * (
                    1.0 - (abs_x[ramp] - w_flat_half) / t_ingress
                )

            return y

        # Initial guesses
        depth_guess = 1.0 - np.min(t_flux)
        total_width = np.ptp(t_phase) / 2.0

        try:
            popt, _ = curve_fit(
                trapezoid,
                t_phase,
                t_flux,
                p0=[depth_guess, total_width * 0.3, total_width * 0.2, 1.0],
                bounds=(
                    [0, 0, 0, 0.9],
                    [0.5, total_width, total_width, 1.1],
                ),
                maxfev=5000,
            )

            _, w_flat_half, t_ingress, _ = popt
            total_dur = w_flat_half + t_ingress

            if total_dur > 0:
                return float(np.clip(w_flat_half / total_dur, 0.0, 1.0))
            return np.nan

        except (RuntimeError, ValueError):
            # Fallback: simple threshold-based estimate
            min_flux = np.min(t_flux)
            max_flux = np.max(t_flux)
            flux_range = max_flux - min_flux

            if flux_range <= 0:
                return np.nan

            threshold = min_flux + 0.1 * flux_range
            flat_mask = t_flux <= threshold

            if np.sum(flat_mask) < 2:
                return 0.0

            flat_width = np.ptp(t_phase[flat_mask])
            total_width_val = np.ptp(t_phase)

            if total_width_val <= 0:
                return np.nan

            return float(np.clip(flat_width / total_width_val, 0.0, 1.0))

    except Exception:
        return np.nan


def _compute_ingress_symmetry(
    phase: np.ndarray,
    flux: np.ndarray,
    half_dur: float,
) -> float:
    """Compute ingress-egress symmetry via Pearson correlation.

    Args:
        phase: Binned phase array.
        flux: Binned flux array.
        half_dur: Half-duration in phase units.

    Returns:
        Pearson correlation coefficient (−1 to 1).
    """
    try:
        from scipy.stats import pearsonr

        # Ingress: negative phase, Egress: positive phase
        ingress_mask = (phase >= -half_dur) & (phase < 0)
        egress_mask = (phase > 0) & (phase <= half_dur)

        ingress = flux[ingress_mask]
        egress = flux[egress_mask]

        if len(ingress) < 3 or len(egress) < 3:
            return np.nan

        # Resample to same length and flip egress
        min_len = min(len(ingress), len(egress))
        ingress_r = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(ingress)),
            ingress,
        )
        egress_r = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(egress)),
            egress[::-1],
        )

        corr, _ = pearsonr(ingress_r, egress_r)
        return float(corr) if np.isfinite(corr) else np.nan

    except Exception:
        return np.nan
