"""
ASTRA Phase Fold — Phase-fold transit light curves for analysis.

Computes orbital phase, stacks all transit repeats, and bins the
folded light curve at optimal resolution (duration / 50).
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.extraction.phase_fold")


class PhaseFoldResult:
    """Result of phase folding a transit light curve.

    Attributes:
        phase: Unbinned phase array (−0.5 to 0.5), transit centered at 0.
        flux: Unbinned phase-folded flux array.
        flux_err: Unbinned flux uncertainty array (phase-folded).
        binned_phase: Binned phase array.
        binned_flux: Binned flux array (mean in each bin).
        binned_flux_err: Binned flux uncertainty (standard error of mean).
        n_in_transit: Number of in-transit data points.
        n_transits_stacked: Number of transit events stacked.
    """

    def __init__(self):
        self.phase: np.ndarray = np.array([])
        self.flux: np.ndarray = np.array([])
        self.flux_err: np.ndarray = np.array([])
        self.binned_phase: np.ndarray = np.array([])
        self.binned_flux: np.ndarray = np.array([])
        self.binned_flux_err: np.ndarray = np.array([])
        self.n_in_transit: int = 0
        self.n_transits_stacked: int = 0


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_bins: int | None = None,
) -> PhaseFoldResult:
    """Phase-fold a light curve at the given period and epoch.

    Phase computation:
        phase = ((time - T0) / period) mod 1
        Center transit at phase 0 by mapping values > 0.5 to value − 1.

    Stacking: For a 27-day sector with a 3-day period, 9 transits are
    stacked. The effective SNR improvement scales as √(n_transits).

    Binning: Default bin width is approximately duration / 50 in phase
    units, providing sufficient resolution to resolve ingress, flat
    bottom, and egress without over-smoothing.

    Args:
        time: Time array in days (BTJD).
        flux: Detrended normalized flux array.
        flux_err: Flux uncertainty array.
        period: Orbital period in days.
        t0: Transit mid-time in days (BTJD).
        duration: Transit duration in days.
        n_bins: Number of phase bins (default: auto-computed from duration).

    Returns:
        PhaseFoldResult with unbinned and binned phase-folded data.
    """
    result = PhaseFoldResult()

    if np.isnan(period) or period <= 0 or np.isnan(t0):
        logger.warning("Invalid period or T0 for phase folding")
        return result

    # Compute phase centered at transit (phase 0)
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # Range: [-0.5, 0.5]

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    folded_flux = flux[sort_idx]
    folded_err = flux_err[sort_idx]

    result.phase = phase
    result.flux = folded_flux
    result.flux_err = folded_err

    # Count in-transit points and stacked transits
    if duration > 0 and not np.isnan(duration):
        half_dur_phase = (duration / period) / 2.0
        in_transit = np.abs(phase) <= half_dur_phase
        result.n_in_transit = int(np.sum(in_transit))

        time_span = time[-1] - time[0]
        result.n_transits_stacked = max(1, int(time_span / period))

    # Compute binned phase fold
    if n_bins is None:
        # Optimal bin size: target roughly 1/50th of the transit duration
        # as per the spec. This provides ~50 bins across the transit.
        if duration > 0 and not np.isnan(duration):
            bin_width_phase = (duration / period) / 50.0
            n_bins = max(50, int(1.0 / bin_width_phase))
            n_bins = min(n_bins, 1000)  # Cap at 1000 bins
        else:
            n_bins = 200  # Fallback default

    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    binned_flux = np.full(n_bins, np.nan)
    binned_err = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            binned_flux[i] = np.nanmean(folded_flux[in_bin])
            # Standard error of the mean
            binned_err[i] = np.nanstd(folded_flux[in_bin]) / np.sqrt(n_in_bin)

    # Remove empty bins
    valid = np.isfinite(binned_flux)
    result.binned_phase = bin_centers[valid]
    result.binned_flux = binned_flux[valid]
    result.binned_flux_err = binned_err[valid]

    logger.debug(
        f"Phase fold: period={period:.4f}d, {result.n_transits_stacked} transits "
        f"stacked, {result.n_in_transit} in-transit points, "
        f"{len(result.binned_phase)} bins"
    )

    return result


def resample_phase_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    target_length: int = 256,
) -> np.ndarray:
    """Resample phase-folded flux to a fixed-length array for CNN input.

    Uses linear interpolation to produce a uniform-length representation
    of the phase-folded light curve. Normalizes to zero mean and unit
    variance.

    Args:
        phase: Phase array (−0.5 to 0.5).
        flux: Phase-folded flux array.
        target_length: Target array length (default 256 for CNN).

    Returns:
        Resampled, normalized flux array of shape (target_length,).
    """
    if len(phase) < 3:
        return np.zeros(target_length)

    # Create uniform phase grid
    target_phase = np.linspace(-0.5, 0.5, target_length)

    # Linear interpolation
    resampled = np.interp(target_phase, phase, flux)

    # Normalize to zero mean, unit variance
    mean = np.nanmean(resampled)
    std = np.nanstd(resampled)

    if std > 0:
        resampled = (resampled - mean) / std
    else:
        resampled = resampled - mean

    # Replace any NaN with 0
    resampled = np.nan_to_num(resampled, nan=0.0)

    return resampled
