"""
ASTRA Downloader — TESS light curve download from MAST archive.

Uses lightkurve to query and download FITS files for a specified TESS sector
with 2-minute (short) cadence data. Extracts stellar parameters from FITS
headers with fallback to TIC catalogue queries.
"""

import time
from pathlib import Path
from typing import Any

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits

from astra.utils.logger import get_logger

logger = get_logger("astra.ingestion.downloader")


class TESSDownloader:
    """Downloads and manages TESS light curve FITS files from the MAST archive.

    Args:
        raw_dir: Directory to store downloaded FITS files.
        cadence: Cadence type ('short' for 2-min, 'long' for 30-min).
        mission: Mission name (default 'TESS').
        max_stars: Maximum number of stars to download per sector.
        download_retries: Number of retries for failed downloads.
        retry_backoff_base: Base for exponential backoff in seconds.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw/",
        cadence: str = "short",
        mission: str = "TESS",
        max_stars: int = 30000,
        download_retries: int = 3,
        retry_backoff_base: float = 2.0,
    ):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cadence = cadence
        self.mission = mission
        self.max_stars = max_stars
        self.download_retries = download_retries
        self.retry_backoff_base = retry_backoff_base

    def search_sector(self, sector: int) -> lk.SearchResult:
        """Search MAST for all light curves in a TESS sector.

        Args:
            sector: TESS sector number.

        Returns:
            lightkurve SearchResult containing available light curves.
        """
        logger.info(
            f"Searching MAST for sector {sector}, "
            f"cadence={self.cadence}, mission={self.mission}"
        )

        # Search for all targets in the sector
        search_result = lk.search_lightcurve(
            f"sector {sector}",
            mission=self.mission,
            cadence=self.cadence,
            author="SPOC",  # Science Processing Operations Center pipeline
        )

        n_results = len(search_result)
        logger.info(f"Found {n_results} light curves in sector {sector}")

        if n_results > self.max_stars:
            logger.info(f"Limiting to {self.max_stars} stars (max_stars config)")
            search_result = search_result[: self.max_stars]

        return search_result

    def download_light_curve(
        self, search_row: Any, sector: int
    ) -> tuple[Path | None, dict]:
        """Download a single light curve with retry logic.

        Args:
            search_row: A single row from a lightkurve SearchResult.
            sector: TESS sector number.

        Returns:
            Tuple of (file_path, metadata_dict). file_path is None on failure.
        """
        tic_id = None
        metadata: dict[str, Any] = {}

        for attempt in range(self.download_retries):
            try:
                # Download the light curve
                lc = search_row.download()

                if lc is None:
                    logger.warning("Download returned None, skipping")
                    return None, metadata

                # Extract TIC ID from the target name
                target_name = str(lc.meta.get("TARGETID", lc.meta.get("OBJECT", "")))
                tic_id = self._extract_tic_id(target_name, lc.meta)

                # Build file path
                fits_path = self.raw_dir / f"TIC_{tic_id}_{sector}.fits"

                # Save the light curve as FITS
                lc.to_fits(
                    path=str(fits_path),
                    overwrite=True,
                )

                # Extract metadata from the light curve object and FITS header
                metadata = self._extract_metadata(lc, tic_id, fits_path, sector)

                logger.debug(f"Downloaded TIC {tic_id} to {fits_path}")
                return fits_path, metadata

            except Exception as e:
                wait_time = self.retry_backoff_base ** (attempt + 1)
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.download_retries} "
                    f"failed for TIC {tic_id}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        logger.error(
            f"All {self.download_retries} download attempts failed for TIC {tic_id}"
        )
        return None, metadata

    def download_sector(self, sector: int) -> pd.DataFrame:
        """Download all light curves for a TESS sector.

        Args:
            sector: TESS sector number.

        Returns:
            DataFrame indexed by TIC_ID with columns:
            file_path, ra, dec, tmag, teff, r_star, crowdsap.
        """
        search_result = self.search_sector(sector)

        metadata_records: list[dict] = []
        n_success = 0
        n_failed = 0

        for i, row in enumerate(search_result):
            file_path, metadata = self.download_light_curve(row, sector)

            if file_path is not None:
                metadata["file_path"] = str(file_path)
                metadata_records.append(metadata)
                n_success += 1
            else:
                n_failed += 1

            # Log progress every 100 stars
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Download progress: {i + 1}/{len(search_result)} "
                    f"({n_success} success, {n_failed} failed)"
                )

        logger.info(
            f"Sector {sector} download complete: "
            f"{n_success} succeeded, {n_failed} failed"
        )

        if not metadata_records:
            logger.warning("No light curves downloaded successfully")
            return pd.DataFrame()

        df = pd.DataFrame(metadata_records)
        if "tic_id" in df.columns:
            df = df.set_index("tic_id")

        # Save metadata CSV
        meta_path = self.raw_dir / f"sector_{sector}_metadata.csv"
        df.to_csv(meta_path)
        logger.info(f"Metadata saved to {meta_path}")

        return df

    def _extract_tic_id(self, target_name: str, meta: dict) -> int:
        """Extract numeric TIC ID from target name or metadata.

        Args:
            target_name: Target name string (e.g., 'TIC 123456789').
            meta: FITS header metadata dictionary.

        Returns:
            Integer TIC ID.
        """
        # Try TICID keyword first
        tic_id = meta.get("TICID", None)
        if tic_id is not None:
            return int(tic_id)

        # Try parsing from target name
        import re

        match = re.search(r"(\d{5,})", str(target_name))
        if match:
            return int(match.group(1))

        # Try TARGETID
        target_id = meta.get("TARGETID", None)
        if target_id is not None:
            return int(target_id)

        raise ValueError(f"Cannot extract TIC ID from target: {target_name}")

    def _extract_metadata(
        self, lc: Any, tic_id: int, fits_path: Path, sector: int
    ) -> dict[str, Any]:
        """Extract stellar parameters from light curve metadata and FITS headers.

        Attempts to read Teff, R_star from FITS header keywords. If absent,
        marks as NaN for later fallback via TIC catalogue query.

        Args:
            lc: lightkurve LightCurve object.
            tic_id: TIC ID of the star.
            fits_path: Path to the saved FITS file.
            sector: TESS sector number.

        Returns:
            Dictionary of extracted metadata.
        """
        meta = lc.meta

        # Extract parameters with NaN defaults
        teff = meta.get("TEFF", np.nan)
        r_star = meta.get("RADIUS", np.nan)
        ra = meta.get("RA_OBJ", meta.get("RA", np.nan))
        dec = meta.get("DEC_OBJ", meta.get("DEC", np.nan))
        tmag = meta.get("TESSMAG", np.nan)
        crowdsap = meta.get("CROWDSAP", np.nan)
        logg = meta.get("LOGG", np.nan)

        # Convert to float, handling None and string values
        metadata = {
            "tic_id": int(tic_id),
            "sector": int(sector),
            "ra": self._safe_float(ra),
            "dec": self._safe_float(dec),
            "tmag": self._safe_float(tmag),
            "teff": self._safe_float(teff),
            "r_star": self._safe_float(r_star),
            "logg": self._safe_float(logg),
            "crowdsap": self._safe_float(crowdsap),
        }

        return metadata

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert a value to float, returning NaN on failure.

        Args:
            value: Value to convert.

        Returns:
            Float value or NaN.
        """
        try:
            if value is None:
                return np.nan
            result = float(value)
            return result if np.isfinite(result) else np.nan
        except (ValueError, TypeError):
            return np.nan

    def query_tic_fallback(self, tic_id: int) -> dict[str, float]:
        """Query the TIC catalogue for stellar parameters as a fallback.

        Used when FITS header keywords are missing for Teff and R_star.

        Args:
            tic_id: TIC ID to query.

        Returns:
            Dictionary with teff and r_star values.
        """
        try:
            from astroquery.mast import Catalogs

            result = Catalogs.query_object(
                f"TIC {tic_id}", catalog="TIC", radius=0.001
            )

            if result is not None and len(result) > 0:
                row = result[0]
                return {
                    "teff": self._safe_float(row.get("Teff", np.nan)),
                    "r_star": self._safe_float(row.get("rad", np.nan)),
                    "logg": self._safe_float(row.get("logg", np.nan)),
                }
        except Exception as e:
            logger.warning(f"TIC catalogue query failed for TIC {tic_id}: {e}")

        return {"teff": np.nan, "r_star": np.nan, "logg": np.nan}
