"""
ASTRA BLS Search — Box Least Squares periodic transit search.

Uses astropy's BoxLeastSquares to search for box-shaped periodic dips
in the detrended flux. The period grid covers 0.5-13.0 days with
19,500 logarithmically-spaced trial periods.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares

from astra.utils.logger import get_logger

logger = get_logger("astra.detection.bls_search")


class BLSResult:
    """Result of a BLS transit search.

    Attributes:
        period: Best period in days.
        t0: Best epoch (transit mid-time) in BTJD.
        duration: Best transit duration in days.
        depth: Transit depth (fractional flux decrease).
        power: BLS power at the best period.
        snr: Signal-to-noise ratio of the detection.
        fap: False alarm probability (Bonferroni-corrected).
        periods: Full period grid searched.
        powers: BLS power at each trial period.
        n_transits: Estimated number of transit events.
        harmonic_flags: Dictionary of harmonic analysis results.
    """

    def __init__(self):
        self.period: float = np.nan
        self.t0: float = np.nan
        self.duration: float = np.nan
        self.depth: float = np.nan
        self.power: float = np.nan
        self.snr: float = 0.0
        self.fap: float = 1.0
        self.periods: np.ndarray = np.array([])
        self.powers: np.ndarray = np.array([])
        self.n_transits: int = 0
        self.harmonic_flags: dict = {}


def run_bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    period_min: float = 0.5,
    period_max: float = 13.0,
    n_trial_periods: int = 19500,
    duration_min_fraction: float = 0.05,
    duration_max_fraction: float = 0.5,
) -> BLSResult:
    """Run Box Least Squares transit search.

    The minimum period of 0.5 days avoids the tidal destruction zone
    (planets closer than 0.5-day orbits are physically unstable).
    The maximum period is half the sector duration (~13 days) to
    guarantee at least 2 observed transit events.

    Args:
        time: Detrended time array in days (BTJD).
        flux: Detrended, normalized flux array.
        flux_err: Flux uncertainty array (optional).
        period_min: Minimum period to search in days.
        period_max: Maximum period to search in days.
        n_trial_periods: Number of trial periods in the grid.
        duration_min_fraction: Minimum duration as fraction of period.
        duration_max_fraction: Maximum duration as fraction of period.

    Returns:
        BLSResult with best period, depth, power, and periodogram.
    """
    result = BLSResult()

    if len(time) < 50:
        logger.warning("Too few data points for BLS search")
        return result

    try:
        # Build the BLS model
        if flux_err is not None and np.all(np.isfinite(flux_err)):
            model = BoxLeastSquares(time, flux, dy=flux_err)
        else:
            model = BoxLeastSquares(time, flux)

        # Period grid: logarithmically spaced for even coverage
        periods = np.logspace(
            np.log10(period_min),
            np.log10(period_max),
            n_trial_periods,
        )

        # Duration grid: cover realistic transit durations (~1 hour to ~7 hours).
        # Physical constraint: a planetary transit typically lasts 1-10% of the
        # orbital period. Allowing durations up to 50% of the period causes the
        # algorithm to fit the entire orbit as one giant "transit box", producing
        # fake high-SNR detections at the minimum period boundary.
        # Cap at 15% of period_min and hard-limit at 0.3 days (7.2 hours).
        duration_min_days = max(0.02, period_min * duration_min_fraction)
        duration_max_days = min(0.3, period_min * 0.15)

        # Safety: ensure min < max and max < period_min
        if duration_max_days <= duration_min_days:
            duration_max_days = duration_min_days + 0.01
        if duration_max_days >= period_min:
            duration_max_days = period_min * 0.5 - 0.001

        durations = np.linspace(duration_min_days, duration_max_days, 20)

        # Run the periodogram
        periodogram = model.power(periods, durations)

        # Extract best period
        best_idx = np.argmax(periodogram.power)
        result.period = float(periodogram.period[best_idx])
        result.t0 = float(periodogram.transit_time[best_idx])
        result.duration = float(periodogram.duration[best_idx])
        result.depth = float(periodogram.depth[best_idx])
        result.power = float(periodogram.power[best_idx])

        # Store the full periodogram
        result.periods = np.array(periodogram.period)
        result.powers = np.array(periodogram.power)

        # Estimate number of transits
        time_span = time[-1] - time[0]
        result.n_transits = max(1, int(time_span / result.period))

        # Compute SNR
        in_transit_mask = _get_transit_mask(
            time, result.period, result.t0, result.duration
        )
        n_in_transit = int(np.sum(in_transit_mask))
        rms_oot = np.nanstd(flux[~in_transit_mask])

        if rms_oot > 0 and result.depth > 0:
            result.snr = result.depth * np.sqrt(max(1, n_in_transit)) / rms_oot
        else:
            result.snr = 0.0

        # Compute FAP with Bonferroni correction
        # FAP_corrected = 1 - (1 - FAP_raw)^n_trial_periods
        try:
            fap_raw = model.false_alarm_probability(
                result.power, method="bootstrap", n_bootstraps=100
            )
            result.fap = 1.0 - (1.0 - fap_raw) ** n_trial_periods
            result.fap = min(1.0, max(0.0, result.fap))
        except Exception:
            # FAP computation can fail; set to 1.0 as conservative default
            result.fap = 1.0

        # Harmonic analysis
        result.harmonic_flags = _analyze_harmonics(
            result.periods, result.powers, result.period
        )

        logger.debug(
            f"BLS: P={result.period:.4f}d, depth={result.depth:.5f}, "
            f"SNR={result.snr:.1f}, power={result.power:.4f}, "
            f"n_transits={result.n_transits}"
        )

    except Exception as e:
        logger.error(f"BLS search failed: {e}", exc_info=True)

    return result


def _get_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> np.ndarray:
    """Create a boolean mask identifying in-transit data points.

    Args:
        time: Time array.
        period: Orbital period in days.
        t0: Transit mid-time.
        duration: Transit duration in days.

    Returns:
        Boolean mask (True = in transit).
    """
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # Center at 0

    half_dur_phase = (duration / period) / 2.0
    return np.abs(phase) <= half_dur_phase


def _analyze_harmonics(
    periods: np.ndarray,
    powers: np.ndarray,
    best_period: float,
    harmonic_threshold: float = 0.5,
) -> dict:
    """Analyze the BLS periodogram for harmonic signatures.

    Periodogram interpretation:
    - Single dominant spike: strong periodic signal
    - Multiple harmonically related spikes (P/2, P/3, 2P, 3P): single
      true signal with aliases
    - Flat forest of equal peaks: no significant signal
    - Two unrelated spikes at non-harmonic periods: potential multi-planet
      system or planet + eclipsing binary

    Args:
        periods: Period grid from BLS.
        powers: Power values from BLS.
        best_period: Best detected period.
        harmonic_threshold: Fraction of peak power to consider significant.

    Returns:
        Dictionary with harmonic analysis results.
    """
    if len(powers) == 0 or np.isnan(best_period):
        return {"type": "NO_SIGNAL", "harmonics_detected": []}

    peak_power = np.max(powers)
    if peak_power <= 0:
        return {"type": "NO_SIGNAL", "harmonics_detected": []}

    # Check for harmonic ratios
    harmonic_ratios = [0.5, 1.0 / 3.0, 2.0, 3.0]
    harmonic_labels = ["P/2", "P/3", "2P", "3P"]
    detected_harmonics: list[str] = []

    for ratio, label in zip(harmonic_ratios, harmonic_labels):
        harmonic_period = best_period * ratio

        if harmonic_period < periods[0] or harmonic_period > periods[-1]:
            continue

        # Find nearest period in grid
        idx = np.argmin(np.abs(periods - harmonic_period))
        harmonic_power = powers[idx]

        if harmonic_power > harmonic_threshold * peak_power:
            detected_harmonics.append(label)

    # Classify the periodogram pattern
    median_power = np.nanmedian(powers)
    if peak_power < 2.0 * median_power:
        pattern = "FLAT_FOREST"  # No significant signal
    elif len(detected_harmonics) > 0:
        pattern = "HARMONIC_ALIASES"  # Single signal with aliases
    else:
        # Check for secondary non-harmonic peaks
        sorted_powers = np.sort(powers)[::-1]
        if len(sorted_powers) > 1 and sorted_powers[1] > 0.7 * peak_power:
            pattern = "MULTI_PEAK"  # Possible multi-planet or planet + EB
        else:
            pattern = "SINGLE_DOMINANT"  # Strong periodic signal

    return {
        "type": pattern,
        "harmonics_detected": detected_harmonics,
        "peak_to_median_ratio": float(peak_power / max(median_power, 1e-10)),
    }
