"""
ASTRA TLS Search — Transit Least Squares periodic transit search.

Uses the transitleastsquares library which fits a physically realistic
limb-darkened transit template (Mandel-Agol) rather than a flat box.
This provides higher sensitivity to small planets compared to BLS.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.detection.tls_search")


class TLSResult:
    """Result of a TLS transit search.

    Attributes:
        period: Best period in days.
        t0: Best epoch (transit mid-time) in BTJD.
        duration: Transit duration in days.
        depth: Transit depth (1 - depth_mean for TLS convention).
        sde: Signal Detection Efficiency score.
        snr: Estimated signal-to-noise ratio.
        transit_times: Array of individual transit mid-times.
        folded_phase: Phase-folded phase array from TLS.
        folded_flux: Phase-folded flux array from TLS.
        n_transits: Number of individual transit events.
        odd_even_mismatch: TLS odd-even depth mismatch metric.
    """

    def __init__(self):
        self.period: float = np.nan
        self.t0: float = np.nan
        self.duration: float = np.nan
        self.depth: float = np.nan
        self.sde: float = 0.0
        self.snr: float = 0.0
        self.transit_times: np.ndarray = np.array([])
        self.folded_phase: np.ndarray = np.array([])
        self.folded_flux: np.ndarray = np.array([])
        self.n_transits: int = 0
        self.odd_even_mismatch: float = np.nan


def run_tls_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    r_star: float = 1.0,
    m_star: float = 1.0,
    teff: float = 5778.0,
    period_min: float = 0.5,
    period_max: float = 13.0,
) -> TLSResult:
    """Run Transit Least Squares search with limb-darkened templates.

    TLS uses physically motivated transit shapes that include limb
    darkening effects, providing higher sensitivity to small planets
    compared to BLS. SDE > 9 is considered a strong signal.

    Args:
        time: Detrended time array in days (BTJD).
        flux: Detrended, normalized flux array.
        flux_err: Flux uncertainty array (optional).
        r_star: Stellar radius in solar radii (for transit template).
        m_star: Stellar mass in solar masses.
        teff: Stellar effective temperature in Kelvin.
        period_min: Minimum period to search in days.
        period_max: Maximum period to search in days.

    Returns:
        TLSResult with best period, SDE, transit times, and folded data.
    """
    result = TLSResult()

    if len(time) < 50:
        logger.warning("Too few data points for TLS search")
        return result

    try:
        from transitleastsquares import transitleastsquares

        # Handle NaN stellar parameters
        r_star = r_star if np.isfinite(r_star) and r_star > 0 else 1.0
        m_star = m_star if np.isfinite(m_star) and m_star > 0 else 1.0

        # Convert to plain numpy arrays (astropy masked arrays cause
        # "cannot write to unmasked output" errors with in-place ops)
        time_np = np.asarray(time, dtype=float)
        flux_np = np.asarray(flux, dtype=float)

        # Remove any remaining NaN/inf values
        mask = np.isfinite(time_np) & np.isfinite(flux_np)
        if flux_err is not None:
            flux_err_np = np.asarray(flux_err, dtype=float)
            mask = mask & np.isfinite(flux_err_np)
        time_clean = time_np[mask]
        flux_clean = flux_np[mask]

        if len(time_clean) < 50:
            logger.warning("Too few finite points for TLS search")
            return result

        # Initialize TLS
        model = transitleastsquares(time_clean, flux_clean)

        # Run the TLS search
        # Pass stellar parameters for physically motivated transit shapes.
        # Explicitly pass M_star_max >= M_star to avoid TLS validation error
        # when catalog stellar mass exceeds TLS internal defaults.
        tls_results = model.power(
            period_min=period_min,
            period_max=period_max,
            R_star=r_star,
            M_star=m_star,
            M_star_max=max(m_star, 1.0),
            show_progress_bar=False,
            use_threads=1,  # Avoid nested parallelism
        )

        # Extract results
        result.period = float(tls_results.period)
        result.t0 = float(tls_results.T0)
        result.duration = float(tls_results.duration)
        result.sde = float(tls_results.SDE)

        # Depth: TLS reports depth as the fractional flux decrease
        if hasattr(tls_results, "depth"):
            result.depth = float(1.0 - tls_results.depth)
        else:
            result.depth = np.nan

        # Transit times for odd-even analysis
        if hasattr(tls_results, "transit_times"):
            result.transit_times = np.array(tls_results.transit_times)
            result.n_transits = len(result.transit_times)
        else:
            # Estimate transit times from period and T0
            time_span = time_clean[-1] - time_clean[0]
            n_transits = max(1, int(time_span / result.period))
            result.transit_times = result.t0 + result.period * np.arange(n_transits)
            result.n_transits = n_transits

        # Phase-folded data
        if hasattr(tls_results, "folded_phase"):
            result.folded_phase = np.array(tls_results.folded_phase)
        if hasattr(tls_results, "folded_y"):
            result.folded_flux = np.array(tls_results.folded_y)

        # Odd-even mismatch from TLS
        if hasattr(tls_results, "odd_even_mismatch"):
            result.odd_even_mismatch = float(tls_results.odd_even_mismatch)

        # Compute SNR estimate from SDE
        # SDE is roughly proportional to SNR but on a different scale
        # For approximate conversion: SNR ≈ SDE × 0.8 for typical cases
        result.snr = result.sde * 0.8

        logger.debug(
            f"TLS: P={result.period:.4f}d, SDE={result.sde:.1f}, "
            f"depth={result.depth:.5f}, duration={result.duration:.4f}d, "
            f"n_transits={result.n_transits}"
        )

    except ImportError:
        logger.error(
            "transitleastsquares not installed. Install with: "
            "pip install transitleastsquares"
        )
    except Exception as e:
        logger.error(f"TLS search failed: {e}", exc_info=True)

    return result
