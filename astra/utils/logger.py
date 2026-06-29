"""
ASTRA Logger — Centralized logging system.

Provides stage-level loggers with timestamped file output and console output.
Logging configuration is loaded from config/logging_config.yaml.
"""

import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path

import yaml


_INITIALIZED = False


def setup_logging(
    config_path: str = "config/logging_config.yaml",
    output_dir: str = "outputs/logs",
    log_level: str = "DEBUG",
) -> None:
    """Initialize the logging system from the YAML configuration.

    Creates the log output directory if it doesn't exist, loads the YAML
    config, patches file handler paths with the output directory, and
    applies the configuration.

    Args:
        config_path: Path to the logging YAML configuration file.
        output_dir: Directory where log files will be written.
        log_level: Default logging level override.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Ensure log directory exists
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            log_config = yaml.safe_load(f)

        # Patch file handler paths with actual output directory and timestamp
        for handler_name, handler_cfg in log_config.get("handlers", {}).items():
            if "filename" in handler_cfg:
                base_name = Path(handler_cfg["filename"]).stem
                handler_cfg["filename"] = str(
                    log_dir / f"{base_name}_{timestamp}.log"
                )

        logging.config.dictConfig(log_config)
    else:
        # Fallback: basic configuration if YAML not found
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.DEBUG),
            format="%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    str(log_dir / f"astra_pipeline_{timestamp}.log"),
                    encoding="utf-8",
                ),
            ],
        )

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a specific pipeline stage.

    If the logging system hasn't been initialized, it will be initialized
    with default settings.

    Args:
        name: Logger name, typically the module path (e.g., 'astra.detection').

    Returns:
        Configured logging.Logger instance.
    """
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


class StageLogger:
    """Context manager for logging pipeline stage execution.

    Tracks stage start/end times, star counts, and error counts.
    Provides structured logging for pipeline progress tracking.

    Usage:
        with StageLogger("preprocessing", logger) as stage:
            stage.log_progress(stars_processed=100, tces_found=5)
    """

    def __init__(self, stage_name: str, logger: logging.Logger | None = None):
        self.stage_name = stage_name
        self.logger = logger or get_logger(f"astra.{stage_name}")
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.stats: dict = {
            "stars_processed": 0,
            "tces_found": 0,
            "candidates_promoted": 0,
            "errors": 0,
        }

    def __enter__(self) -> "StageLogger":
        self.start_time = datetime.now()
        self.logger.info(
            "=" * 60 + f"\n  STAGE: {self.stage_name.upper()} — STARTED"
            f"\n  Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 60
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        status = "COMPLETED" if exc_type is None else "FAILED"

        self.logger.info(
            "=" * 60
            + f"\n  STAGE: {self.stage_name.upper()} — {status}"
            f"\n  Duration: {elapsed:.1f}s"
            f"\n  Stars processed: {self.stats['stars_processed']}"
            f"\n  TCEs found: {self.stats['tces_found']}"
            f"\n  Candidates promoted: {self.stats['candidates_promoted']}"
            f"\n  Errors: {self.stats['errors']}\n"
            + "=" * 60
        )

        if exc_type is not None:
            self.logger.error(
                f"Stage {self.stage_name} failed with {exc_type.__name__}: {exc_val}",
                exc_info=True,
            )

        # Don't suppress exceptions
        return False

    def log_progress(self, **kwargs) -> None:
        """Update stage statistics and log progress.

        Args:
            **kwargs: Key-value pairs to update in the stats dictionary.
                      Common keys: stars_processed, tces_found,
                      candidates_promoted, errors.
        """
        self.stats.update(kwargs)
        self.logger.debug(
            f"[{self.stage_name}] Progress: "
            + ", ".join(f"{k}={v}" for k, v in self.stats.items())
        )

    def log_star_error(self, tic_id: int, error: Exception) -> None:
        """Log an error for a specific star without crashing the pipeline.

        Args:
            tic_id: TIC ID of the star that caused the error.
            error: The exception that was raised.
        """
        self.stats["errors"] += 1
        self.logger.error(
            f"[{self.stage_name}] Error processing TIC {tic_id}: "
            f"{type(error).__name__}: {error}",
            exc_info=True,
        )
