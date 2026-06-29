"""
ASTRA Vetting — Centroid Shift Analysis (Test 03).

Detects background eclipsing binaries (BEBs) by measuring the flux-weighted
centroid shift during transit in the Target Pixel File. A shift indicates
the transit signal originates from a different star blended into the TESS pixel.

HARD REJECTION: sigma > threshold → BACKGROUND_ECLIPSING_BINARY
If TPF is not available, the test is skipped (centroid_shift = NaN).
"""

import numpy as np

from astra.utils.logger import get_logger
from astra.utils.stellar_utils import TESS_PLATE_SCALE

logger = get_logger("astra.vetting.centroid")


def test_centroid_shift(
    tpf,
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    sigma_threshold: float = 3.0,
) -> tuple[bool, float, float, bool]:
    """Analyze centroid shift during transit using the Target Pixel File.

    Computes flux-weighted centroid positions during in-transit and
    out-of-transit cadences, then measures the shift in arcseconds.

    Physical justification: If the transit signal comes from a background
    EB blended into the TESS pixel, the centroid will shift toward the
    background source during transit (when the background EB dims).

    TESS plate scale: 21 arcsec/pixel.

    Args:
        tpf: lightkurve TargetPixelFile object (or None if unavailable).
        time: Time array for the light curve.
        period: Orbital period in days.
        t0: Transit mid-time in BTJD.
        duration: Transit duration in days.
        sigma_threshold: Significance threshold for hard rejection.

    Returns:
        Tuple of (flag_fired, centroid_shift_arcsec, sigma_centroid, hard_reject).
        Returns (False, NaN, NaN, False) if TPF is not available.
    """
    if tpf is None:
        logger.debug("Centroid test: TPF not available, skipping")
        return False, np.nan, np.nan, False

    try:
        # Get flux data from TPF
        flux_cube = tpf.flux.value  # Shape: (n_cadences, n_rows, n_cols)
        tpf_time = tpf.time.btjd

        if flux_cube.ndim != 3:
            logger.warning("TPF flux has unexpected dimensions, skipping centroid test")
            return False, np.nan, np.nan, False

        n_cadences, n_rows, n_cols = flux_cube.shape

        # Create coordinate grids
        row_coords = np.arange(n_rows)
        col_coords = np.arange(n_cols)
        col_grid, row_grid = np.meshgrid(col_coords, row_coords)

        # Identify in-transit and out-of-transit cadences
        phase = ((tpf_time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        half_dur_phase = (duration / period) / 2.0

        in_transit = np.abs(phase) <= half_dur_phase
        out_of_transit = np.abs(phase) > half_dur_phase * 2.0  # Buffer zone

        if np.sum(in_transit) < 3 or np.sum(out_of_transit) < 10:
            logger.debug("Centroid test: insufficient transit/OOT cadences")
            return False, np.nan, np.nan, False

        # Compute flux-weighted centroids
        # centroid = sum(flux_i × coord_i) / sum(flux_i)

        def compute_centroid(flux_frames):
            """Compute mean flux-weighted centroid from multiple frames."""
            x_centroids = []
            y_centroids = []

            for frame in flux_frames:
                total_flux = np.nansum(frame)
                if total_flux <= 0:
                    continue
                x_c = np.nansum(frame * col_grid) / total_flux
                y_c = np.nansum(frame * row_grid) / total_flux
                x_centroids.append(x_c)
                y_centroids.append(y_c)

            if len(x_centroids) == 0:
                return np.nan, np.nan, np.nan, np.nan

            return (
                np.mean(x_centroids),
                np.mean(y_centroids),
                np.std(x_centroids) / np.sqrt(len(x_centroids)),
                np.std(y_centroids) / np.sqrt(len(y_centroids)),
            )

        # In-transit centroid
        x_in, y_in, sx_in, sy_in = compute_centroid(flux_cube[in_transit])
        # Out-of-transit centroid
        x_out, y_out, sx_out, sy_out = compute_centroid(flux_cube[out_of_transit])

        if np.any(np.isnan([x_in, y_in, x_out, y_out])):
            logger.debug("Centroid test: centroid computation returned NaN")
            return False, np.nan, np.nan, False

        # Centroid shift in pixels
        dx = x_in - x_out
        dy = y_in - y_out
        shift_pixels = np.sqrt(dx**2 + dy**2)

        # Convert to arcseconds
        shift_arcsec = shift_pixels * TESS_PLATE_SCALE

        # Uncertainty
        sigma_shift = np.sqrt(
            (sx_in**2 + sx_out**2) + (sy_in**2 + sy_out**2)
        ) * TESS_PLATE_SCALE

        if sigma_shift <= 0:
            return False, float(shift_arcsec), np.nan, False

        # Significance
        sigma_centroid = shift_arcsec / sigma_shift

        flag_fired = sigma_centroid > sigma_threshold
        hard_reject = flag_fired

        if hard_reject:
            logger.info(
                f"Centroid test FAILED: shift={shift_arcsec:.2f} arcsec, "
                f"σ={sigma_centroid:.2f} > {sigma_threshold} → BACKGROUND_EB"
            )
        else:
            logger.debug(
                f"Centroid test passed: shift={shift_arcsec:.2f} arcsec, "
                f"σ={sigma_centroid:.2f}"
            )

        return flag_fired, float(shift_arcsec), float(sigma_centroid), hard_reject

    except Exception as e:
        logger.warning(f"Centroid test failed with error: {e}")
        return False, np.nan, np.nan, False
