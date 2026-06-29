"""
ASTRA Checkpoint — Save and resume intermediate pipeline results.

Implements checkpointing for the per-star processing loop so the pipeline
can resume without reprocessing already-completed stars.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astra.utils.logger import get_logger

logger = get_logger("astra.utils.checkpoint")


class CheckpointManager:
    """Manages pipeline checkpointing for resumable execution.

    Saves intermediate results (processed TIC IDs, candidate DataFrames,
    feature vectors) to disk after each batch. On resume, loads the
    checkpoint and skips already-processed stars.

    Args:
        checkpoint_dir: Directory to store checkpoint files.
        enabled: Whether checkpointing is active.
        batch_size: Number of stars to process before flushing to disk.
    """

    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints/",
        enabled: bool = True,
        batch_size: int = 1000,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.batch_size = batch_size
        self.processed_tics: set[int] = set()
        self._pending_results: list[dict] = []
        self._sector: int | None = None

        if self.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def set_sector(self, sector: int) -> None:
        """Set the current sector for checkpoint file naming.

        Args:
            sector: TESS sector number.
        """
        self._sector = sector

    def _get_checkpoint_path(self, name: str) -> Path:
        """Get the full path for a checkpoint file.

        Args:
            name: Base name of the checkpoint file.

        Returns:
            Full path to the checkpoint file.
        """
        sector_str = f"sector_{self._sector}" if self._sector else "unknown"
        return self.checkpoint_dir / f"{sector_str}_{name}"

    def load_processed_tics(self) -> set[int]:
        """Load the set of already-processed TIC IDs from checkpoint.

        Returns:
            Set of TIC IDs that have been processed in previous runs.
        """
        if not self.enabled:
            return set()

        tic_path = self._get_checkpoint_path("processed_tics.json")
        if tic_path.exists():
            with open(tic_path, "r", encoding="utf-8") as f:
                self.processed_tics = set(json.load(f))
            logger.info(
                f"Loaded checkpoint: {len(self.processed_tics)} stars already processed"
            )
        else:
            self.processed_tics = set()

        return self.processed_tics

    def save_processed_tics(self) -> None:
        """Save the current set of processed TIC IDs to disk."""
        if not self.enabled:
            return

        tic_path = self._get_checkpoint_path("processed_tics.json")
        with open(tic_path, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_tics), f)

    def mark_processed(self, tic_id: int) -> None:
        """Mark a TIC ID as processed.

        Args:
            tic_id: The TIC ID that has been processed.
        """
        self.processed_tics.add(tic_id)

    def is_processed(self, tic_id: int) -> bool:
        """Check if a TIC ID has already been processed.

        Args:
            tic_id: The TIC ID to check.

        Returns:
            True if the star was already processed.
        """
        return tic_id in self.processed_tics

    def add_result(self, result: dict) -> None:
        """Add a star's processing result to the pending buffer.

        If the buffer reaches batch_size, automatically flush to disk.

        Args:
            result: Dictionary containing the star's processing results.
        """
        if not self.enabled:
            return

        self._pending_results.append(result)

        if len(self._pending_results) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush pending results and processed TIC IDs to disk."""
        if not self.enabled or not self._pending_results:
            return

        # Save results as CSV (append mode)
        results_path = self._get_checkpoint_path("candidates_checkpoint.csv")
        df = pd.DataFrame(self._pending_results)

        if results_path.exists():
            df.to_csv(results_path, mode="a", header=False, index=False)
        else:
            df.to_csv(results_path, index=False)

        logger.info(
            f"Checkpoint flushed: {len(self._pending_results)} results saved "
            f"({len(self.processed_tics)} total stars processed)"
        )

        self._pending_results.clear()
        self.save_processed_tics()

    def load_checkpoint_results(self) -> pd.DataFrame | None:
        """Load previously checkpointed candidate results.

        Returns:
            DataFrame of checkpointed results, or None if no checkpoint exists.
        """
        if not self.enabled:
            return None

        results_path = self._get_checkpoint_path("candidates_checkpoint.csv")
        if results_path.exists():
            df = pd.read_csv(results_path)
            logger.info(f"Loaded {len(df)} checkpointed candidate results")
            return df

        return None

    def save_light_curve(
        self,
        tic_id: int,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        trend: np.ndarray | None = None,
    ) -> None:
        """Save a processed light curve to the processed data directory.

        Args:
            tic_id: TIC ID of the star.
            time: Time array in BTJD.
            flux: Detrended flux array.
            flux_err: Flux uncertainty array.
            trend: Estimated trend array (optional).
        """
        processed_dir = Path("data/processed/")
        processed_dir.mkdir(parents=True, exist_ok=True)

        save_dict: dict[str, Any] = {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
        }
        if trend is not None:
            save_dict["trend"] = trend

        np.savez_compressed(
            processed_dir / f"TIC_{tic_id}_processed.npz",
            **save_dict,
        )

    def load_light_curve(
        self, tic_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Load a previously processed light curve.

        Args:
            tic_id: TIC ID of the star.

        Returns:
            Tuple of (time, flux, flux_err, trend) arrays, or None if not found.
        """
        lc_path = Path("data/processed/") / f"TIC_{tic_id}_processed.npz"
        if not lc_path.exists():
            return None

        data = np.load(lc_path)
        trend = data["trend"] if "trend" in data else None
        return data["time"], data["flux"], data["flux_err"], trend

    def clear(self) -> None:
        """Clear all checkpoint data for the current sector."""
        if not self.enabled:
            return

        for path in self.checkpoint_dir.glob(
            f"sector_{self._sector}_*" if self._sector else "*"
        ):
            path.unlink()

        self.processed_tics.clear()
        self._pending_results.clear()
        logger.info("Checkpoint data cleared")
