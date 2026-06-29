"""
ASTRA Normalization — Flux normalization to unity baseline.

Normalizes the detrended flux so the out-of-transit baseline is 1.0
by dividing by the median of the out-of-transit flux.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.preprocessing.normalization")


def normalize_flux(
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    method: str = "median",
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Normalize flux to a baseline of 1.0.

    Divides the flux by its median (or mean), so that the out-of-transit
    baseline is approximately 1.0. Flux errors are scaled accordingly.

    Args:
        flux: Raw or detrended flux array.
        flux_err: Flux uncertainty array (optional).
        method: Normalization method ('median' or 'mean').

    Returns:
        Tuple of (normalized_flux, normalized_flux_err, normalization_factor).
        normalized_flux_err is None if flux_err was None.
    """
    if method == "median":
        norm_factor = np.nanmedian(flux)
    elif method == "mean":
        norm_factor = np.nanmean(flux)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'median' or 'mean'.")

    if norm_factor <= 0 or not np.isfinite(norm_factor):
        logger.warning(
            f"Invalid normalization factor: {norm_factor}. "
            "Using 1.0 as fallback."
        )
        norm_factor = 1.0

    norm_flux = flux / norm_factor

    norm_err = None
    if flux_err is not None:
        norm_err = flux_err / norm_factor

    logger.debug(f"Normalized flux by factor {norm_factor:.6f} (method={method})")

    return norm_flux, norm_err, float(norm_factor)


def sigma_clip(
    flux: np.ndarray,
    sigma: float = 5.0,
    max_iterations: int = 10,
) -> np.ndarray:
    """Apply iterative sigma clipping to remove outliers.

    Removes flux values beyond `sigma` standard deviations from the
    running median. Iterates until convergence or max_iterations.

    This eliminates residual cosmic ray hits and flare spikes not
    caught by TESS quality flags.

    Args:
        flux: Flux array to clip.
        sigma: Clipping threshold in standard deviations (default 5.0).
        max_iterations: Maximum number of clipping iterations (default 10).

    Returns:
        Boolean mask where True = valid (not clipped) data points.
    """
    mask = np.ones(len(flux), dtype=bool)

    for iteration in range(max_iterations):
        valid_flux = flux[mask]

        if len(valid_flux) == 0:
            break

        median = np.nanmedian(valid_flux)
        std = np.nanstd(valid_flux)

        if std <= 0:
            break

        new_mask = np.abs(flux - median) <= sigma * std
        # Combine with existing mask (can only remove points, not add)
        new_mask = mask & new_mask

        n_clipped = np.sum(mask) - np.sum(new_mask)

        if n_clipped == 0:
            logger.debug(
                f"Sigma clipping converged at iteration {iteration + 1}"
            )
            break

        mask = new_mask

    n_removed = len(flux) - np.sum(mask)
    logger.debug(
        f"Sigma clipping ({sigma}σ): removed {n_removed}/{len(flux)} points "
        f"({100 * n_removed / max(1, len(flux)):.1f}%)"
    )

    return mask
