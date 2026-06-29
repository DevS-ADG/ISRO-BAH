"""
ASTRA Gap Handler — Light curve segmentation at instrumental gaps.

Identifies time gaps larger than a threshold (default 0.5 days) and
segments the light curve so that detrending is applied independently
to each continuous segment, preventing filter edge effects.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.preprocessing.gap_handler")


def find_gaps(
    time: np.ndarray, gap_threshold_days: float = 0.5
) -> list[tuple[int, int]]:
    """Identify continuous segments in a time array.

    A gap is defined as a time difference between consecutive points
    exceeding the gap_threshold_days. Each segment is returned as a
    (start_index, end_index) tuple (inclusive).

    Args:
        time: Sorted time array in days (e.g., BTJD).
        gap_threshold_days: Minimum gap duration to trigger segmentation.

    Returns:
        List of (start_index, end_index) tuples for each continuous segment.
    """
    if len(time) == 0:
        return []

    # Find gap locations
    dt = np.diff(time)
    gap_indices = np.where(dt > gap_threshold_days)[0]

    segments: list[tuple[int, int]] = []
    start = 0

    for gap_idx in gap_indices:
        segments.append((start, gap_idx))  # End index is inclusive
        start = gap_idx + 1

    # Add the final segment
    segments.append((start, len(time) - 1))

    logger.debug(
        f"Found {len(segments)} continuous segments "
        f"({len(gap_indices)} gaps > {gap_threshold_days} days)"
    )

    return segments


def segment_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    gap_threshold_days: float = 0.5,
    min_segment_points: int = 20,
) -> list[dict[str, np.ndarray]]:
    """Segment a light curve at gaps into independent segments.

    Each segment is returned as a dictionary containing time, flux,
    and flux_err arrays for that segment.

    Args:
        time: Time array in days.
        flux: Flux array.
        flux_err: Flux uncertainty array.
        gap_threshold_days: Minimum gap to trigger segmentation.
        min_segment_points: Minimum points for a valid segment.

    Returns:
        List of segment dictionaries with 'time', 'flux', 'flux_err' keys.
    """
    segments_indices = find_gaps(time, gap_threshold_days)
    segments: list[dict[str, np.ndarray]] = []

    for start, end in segments_indices:
        n_points = end - start + 1

        if n_points < min_segment_points:
            logger.debug(
                f"Skipping segment [{start}:{end}] with {n_points} points "
                f"(< {min_segment_points} minimum)"
            )
            continue

        segments.append(
            {
                "time": time[start : end + 1],
                "flux": flux[start : end + 1],
                "flux_err": flux_err[start : end + 1],
                "start_idx": start,
                "end_idx": end,
            }
        )

    logger.debug(
        f"Segmented light curve into {len(segments)} valid segments "
        f"(from {len(segments_indices)} total)"
    )

    return segments


def reassemble_segments(
    segments: list[dict[str, np.ndarray]],
    keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Reassemble segmented arrays back into full-length arrays.

    Args:
        segments: List of segment dictionaries from segment_light_curve.
        keys: Keys to reassemble (default: time, flux, flux_err).

    Returns:
        Dictionary of reassembled arrays.
    """
    if keys is None:
        keys = ["time", "flux", "flux_err"]

    result: dict[str, np.ndarray] = {}

    for key in keys:
        arrays = [seg[key] for seg in segments if key in seg]
        if arrays:
            result[key] = np.concatenate(arrays)
        else:
            result[key] = np.array([])

    return result
