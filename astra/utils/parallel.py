"""
ASTRA Parallel Processing — Multiprocessing wrapper for per-star pipeline.

Wraps the per-star processing loop in a multiprocessing.Pool with
configurable core count and shared result assembly.
"""

import multiprocessing as mp
import traceback
from typing import Any, Callable

from tqdm import tqdm

from astra.utils.logger import get_logger

logger = get_logger("astra.utils.parallel")


def _worker_wrapper(args: tuple) -> dict[str, Any]:
    """Wrapper function for multiprocessing workers.

    Catches all exceptions inside the worker so that a single star's
    failure does not crash the entire pool.

    Args:
        args: Tuple of (worker_function, tic_id, worker_kwargs).

    Returns:
        Dictionary with processing results or error information.
    """
    worker_fn, tic_id, kwargs = args

    try:
        result = worker_fn(tic_id, **kwargs)
        result["status"] = "SUCCESS"
        result["tic_id"] = tic_id
        return result
    except Exception as e:
        return {
            "tic_id": tic_id,
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }


def parallel_process_stars(
    tic_ids: list[int],
    worker_fn: Callable,
    n_cores: int = 4,
    worker_kwargs: dict | None = None,
    desc: str = "Processing stars",
) -> list[dict[str, Any]]:
    """Process multiple stars in parallel using multiprocessing.

    Each worker processes one star independently. Results are collected
    and returned as a list of dictionaries.

    Args:
        tic_ids: List of TIC IDs to process.
        worker_fn: Function that processes a single star.
                   Signature: worker_fn(tic_id: int, **kwargs) -> dict
        n_cores: Number of parallel worker processes.
        worker_kwargs: Additional keyword arguments passed to each worker.
        desc: Description for the progress bar.

    Returns:
        List of result dictionaries, one per star (including failures).
    """
    if worker_kwargs is None:
        worker_kwargs = {}

    n_stars = len(tic_ids)
    logger.info(f"Starting parallel processing: {n_stars} stars on {n_cores} cores")

    # Prepare arguments for each worker
    worker_args = [(worker_fn, tic_id, worker_kwargs) for tic_id in tic_ids]

    results: list[dict[str, Any]] = []
    n_success = 0
    n_failed = 0

    if n_cores <= 1:
        # Sequential processing (useful for debugging)
        for args in tqdm(worker_args, desc=desc, unit="star"):
            result = _worker_wrapper(args)
            results.append(result)
            if result["status"] == "SUCCESS":
                n_success += 1
            else:
                n_failed += 1
                logger.error(
                    f"Star TIC {result['tic_id']} failed: "
                    f"{result.get('error_type', 'Unknown')}: "
                    f"{result.get('error_message', 'No message')}"
                )
    else:
        # Parallel processing with multiprocessing.Pool
        # Use 'spawn' context for cross-platform compatibility
        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=n_cores) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker_wrapper, worker_args),
                total=n_stars,
                desc=desc,
                unit="star",
            ):
                results.append(result)
                if result["status"] == "SUCCESS":
                    n_success += 1
                else:
                    n_failed += 1
                    logger.error(
                        f"Star TIC {result['tic_id']} failed: "
                        f"{result.get('error_type', 'Unknown')}: "
                        f"{result.get('error_message', 'No message')}"
                    )

    logger.info(
        f"Parallel processing complete: {n_success} succeeded, "
        f"{n_failed} failed out of {n_stars} total"
    )

    return results


def sequential_process_stars(
    tic_ids: list[int],
    worker_fn: Callable,
    worker_kwargs: dict | None = None,
    desc: str = "Processing stars",
) -> list[dict[str, Any]]:
    """Process stars sequentially (for debugging and single-star mode).

    Same interface as parallel_process_stars but runs in a single process.

    Args:
        tic_ids: List of TIC IDs to process.
        worker_fn: Function that processes a single star.
        worker_kwargs: Additional keyword arguments passed to each worker.
        desc: Description for the progress bar.

    Returns:
        List of result dictionaries.
    """
    return parallel_process_stars(
        tic_ids=tic_ids,
        worker_fn=worker_fn,
        n_cores=1,
        worker_kwargs=worker_kwargs,
        desc=desc,
    )
