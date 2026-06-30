"""
ASTRA Pipeline Orchestrator — Top-level pipeline runner.

Coordinates all 6 pipeline stages: Ingestion → Preprocessing → Detection →
Extraction & Vetting → Classification → Reporting.

Provides run(), run_single_star(), and resume() interfaces.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from astra.utils.logger import setup_logging, get_logger, StageLogger
from astra.utils.checkpoint import CheckpointManager
from astra.utils.stellar_utils import estimate_stellar_mass

logger = get_logger("astra.pipeline")


@dataclass
class PipelineResult:
    """Result of a full pipeline run.

    Attributes:
        n_stars_processed: Total stars processed.
        n_tces: Total Threshold Crossing Events found.
        n_planet_candidates: Number of planet candidates.
        tier1_candidates: DataFrame of Tier 1 candidates.
        tier2_candidates: DataFrame of Tier 2 candidates.
        tier3_candidates: DataFrame of Tier 3 candidates.
        full_catalogue_path: Path to the full CSV catalogue.
        report_path: Path to the 3-page PDF report.
        plot_directory: Path to the diagnostic plots directory.
    """

    n_stars_processed: int = 0
    n_tces: int = 0
    n_planet_candidates: int = 0
    tier1_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    tier2_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    tier3_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    full_catalogue_path: str = ""
    report_path: str = ""
    plot_directory: str = ""


class ASTRAPipeline:
    """Top-level ASTRA pipeline orchestrator.

    Coordinates all pipeline stages, manages configuration, checkpointing,
    and parallel processing.

    Args:
        config_path: Path to the master config.yaml file.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize logging
        setup_logging(
            config_path="config/logging_config.yaml",
            output_dir=str(Path(self.config["reporting"]["output_dir"]) / "logs"),
        )

        # Initialize checkpoint manager
        self.checkpoint = CheckpointManager(
            checkpoint_dir=self.config["pipeline"]["checkpoint_dir"],
            enabled=self.config["pipeline"]["checkpoint_enabled"],
            batch_size=self.config["pipeline"].get("batch_size", 1000),
        )

        # ML classifiers (lazy-loaded)
        self._rf_classifier = None
        self._xgb_classifier = None
        self._cnn_classifier = None
        self._ensemble = None

    @staticmethod
    def _load_config(path: str) -> dict:
        """Load and validate the YAML configuration.

        Args:
            path: Path to config.yaml.

        Returns:
            Configuration dictionary.
        """
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {path}")
        return config

    def _load_classifiers(self) -> None:
        """Load trained ML classifiers."""
        from astra.classification.rf_classifier import RFClassifier
        from astra.classification.xgb_classifier import XGBMultiClassifier
        from astra.classification.cnn_classifier import CNNClassifier
        from astra.classification.ensemble import EnsembleClassifier

        cls_cfg = self.config["classification"]

        self._rf_classifier = RFClassifier(model_dir="models/rf_phase1/")
        rf_loaded = self._rf_classifier.load()
        if not rf_loaded:
            logger.warning("RF model not found — classification will be skipped")

        self._xgb_classifier = XGBMultiClassifier(model_dir="models/xgboost_phase2/")
        xgb_loaded = self._xgb_classifier.load()
        if not xgb_loaded:
            logger.warning("XGBoost model not found — classification will be skipped")

        self._cnn_classifier = CNNClassifier(model_dir="models/cnn_phase2/")
        cnn_loaded = self._cnn_classifier.load()
        if not cnn_loaded:
            logger.warning("CNN model not found — will use XGBoost only")

        self._ensemble = EnsembleClassifier(
            weight_xgb=cls_cfg.get("ensemble_weight_xgb", 0.6),
            weight_cnn=cls_cfg.get("ensemble_weight_cnn", 0.4),
            confidence_planet_threshold=cls_cfg.get("confidence_planet_threshold", 0.80),
            confidence_marginal_threshold=cls_cfg.get("confidence_marginal_threshold", 0.50),
        )

    def run(self, sector: int) -> PipelineResult:
        """Run the full pipeline on a TESS sector.

        Args:
            sector: TESS sector number.

        Returns:
            PipelineResult with all outputs.
        """
        logger.info(f"{'='*60}")
        logger.info(f"  ASTRA Pipeline — Sector {sector}")
        logger.info(f"{'='*60}")

        result = PipelineResult()
        self.checkpoint.set_sector(sector)

        # ── Stage 1: Ingestion ──────────────────────────────────────────
        with StageLogger("ingestion") as stage:
            metadata_df = self._run_ingestion(sector)
            stage.log_progress(stars_processed=len(metadata_df))

        # ── Stage 2: Per-star processing ────────────────────────────────
        # (Preprocessing → Detection → Extraction → Vetting)
        all_candidates: list[dict] = []
        n_stars_quality = 0

        with StageLogger("processing") as stage:
            for i, (tic_id, star_meta) in enumerate(metadata_df.iterrows()):
                if self.checkpoint.is_processed(int(tic_id)):
                    continue

                try:
                    candidates = self._process_single_star(
                        int(tic_id), star_meta, sector
                    )
                    all_candidates.extend(candidates)

                    self.checkpoint.mark_processed(int(tic_id))
                    for cand in candidates:
                        self.checkpoint.add_result(cand)

                    n_stars_quality += 1
                    stage.log_progress(
                        stars_processed=i + 1,
                        tces_found=len(all_candidates),
                    )

                except Exception as e:
                    stage.log_star_error(int(tic_id), e)

            self.checkpoint.flush()

        result.n_stars_processed = len(metadata_df)
        result.n_tces = len(all_candidates)

        # ── Save features and CNN inputs for training ────────────────────
        self._save_training_data(all_candidates)

        # ── Stage 3: Classification ─────────────────────────────────────
        with StageLogger("classification") as stage:
            all_candidates = self._run_classification(all_candidates)
            stage.log_progress(
                candidates_promoted=sum(
                    1 for c in all_candidates
                    if c.get("tier") in ("tier1", "tier2", "tier3")
                )
            )

        # ── Stage 4: Reporting ──────────────────────────────────────────
        with StageLogger("reporting") as stage:
            result = self._run_reporting(
                all_candidates, result, sector, n_stars_quality
            )

        logger.info(f"{'='*60}")
        logger.info(
            f"  ASTRA Pipeline Complete — Sector {sector}\n"
            f"  Stars: {result.n_stars_processed}, TCEs: {result.n_tces}, "
            f"  Candidates: {result.n_planet_candidates}"
        )
        logger.info(f"{'='*60}")

        return result

    def run_single_star(self, tic_id: int, sector: int | None = None) -> list[dict]:
        """Run the pipeline on a single star for testing/debugging.

        Args:
            tic_id: TIC ID of the star.
            sector: TESS sector number (optional).

        Returns:
            List of candidate dictionaries.
        """
        logger.info(f"Processing single star: TIC {tic_id}")

        from astra.ingestion.downloader import TESSDownloader

        sector_val = sector if sector is not None else self.config["pipeline"].get("sector")
        ing_cfg = self.config["ingestion"]

        downloader = TESSDownloader(
            cadence=ing_cfg["cadence"],
            mission=ing_cfg["mission"],
        )

        # Download this specific star
        import lightkurve as lk

        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            sector=sector_val,
            cadence=ing_cfg["cadence"],
            author="SPOC",
        )

        if len(search) == 0:
            logger.error(f"No light curve found for TIC {tic_id} in sector {sector_val}")
            return []

        lc = search[0].download()
        actual_sector = search[0].sector if hasattr(search[0], 'sector') else (sector_val or 1)
        if lc is None:
            logger.error(f"Download failed for TIC {tic_id}")
            return []

        # Build metadata
        meta = {
            "teff": float(lc.meta.get("TEFF", np.nan)),
            "r_star": float(lc.meta.get("RADIUS", np.nan)),
            "crowdsap": float(lc.meta.get("CROWDSAP", np.nan)),
        }

        candidates = self._process_single_star(tic_id, meta, actual_sector, lc_obj=lc)

        # Run classification
        self._load_classifiers()
        candidates = self._run_classification(candidates)

        return candidates

    def resume(self, sector: int, checkpoint_path: str | None = None) -> PipelineResult:
        """Resume a previously interrupted pipeline run.

        Args:
            sector: TESS sector number.
            checkpoint_path: Path to checkpoint directory (optional).

        Returns:
            PipelineResult.
        """
        if checkpoint_path:
            self.checkpoint = CheckpointManager(checkpoint_dir=checkpoint_path)

        self.checkpoint.set_sector(sector)
        self.checkpoint.load_processed_tics()

        logger.info(
            f"Resuming sector {sector}: "
            f"{len(self.checkpoint.processed_tics)} stars already processed"
        )

        return self.run(sector)

    def _run_ingestion(self, sector: int) -> pd.DataFrame:
        """Run the ingestion stage."""
        from astra.ingestion.downloader import TESSDownloader

        ing_cfg = self.config["ingestion"]

        downloader = TESSDownloader(
            cadence=ing_cfg["cadence"],
            mission=ing_cfg["mission"],
            max_stars=ing_cfg["max_stars"],
            download_retries=ing_cfg.get("download_retries", 3),
        )

        return downloader.download_sector(sector)

    def _process_single_star(
        self,
        tic_id: int,
        meta: dict | Any,
        sector: int,
        lc_obj: Any = None,
    ) -> list[dict]:
        """Process a single star through preprocessing → detection → vetting.

        Args:
            tic_id: TIC ID.
            meta: Stellar metadata dictionary or pandas Series.
            sector: TESS sector.
            lc_obj: Pre-loaded lightkurve object (optional).

        Returns:
            List of candidate dictionaries.
        """
        from astra.ingestion.quality_filter import apply_quality_filter, filter_from_arrays
        from astra.preprocessing.detrending import detrend_light_curve
        from astra.detection.multi_planet import search_multi_planet
        from astra.extraction.phase_fold import phase_fold, resample_phase_fold
        from astra.extraction.feature_extractor import extract_features, features_to_array
        from astra.vetting import run_all_vetting

        pre_cfg = self.config["preprocessing"]
        det_cfg = self.config["detection"]
        vet_cfg = self.config["vetting"]

        # Get stellar parameters
        if hasattr(meta, "get"):
            teff = float(meta.get("teff", np.nan))
            r_star = float(meta.get("r_star", np.nan))
            crowdsap = float(meta.get("crowdsap", np.nan))
        else:
            teff = float(getattr(meta, "teff", np.nan))
            r_star = float(getattr(meta, "r_star", np.nan))
            crowdsap = float(getattr(meta, "crowdsap", np.nan))

        # Default stellar params if missing
        if np.isnan(teff) or teff <= 0:
            teff = 5778.0
        if np.isnan(r_star) or r_star <= 0:
            r_star = 1.0

        m_star = estimate_stellar_mass(teff, r_star)

        # ── Quality Filter ──────────────────────────────────────────────
        if lc_obj is not None:
            qf_result = apply_quality_filter(
                lc_obj,
                min_datapoints=pre_cfg["min_datapoints"],
                min_crowdsap=pre_cfg["min_crowdsap"],
                max_rms_threshold=pre_cfg["max_rms_threshold"],
            )
        else:
            # Try loading from saved FITS
            import lightkurve as lk

            file_path = (
                meta.get("file_path", "") if hasattr(meta, "get")
                else getattr(meta, "file_path", "")
            )
            if file_path and Path(file_path).exists():
                lc_obj = lk.read(file_path)
                qf_result = apply_quality_filter(
                    lc_obj,
                    min_datapoints=pre_cfg["min_datapoints"],
                    min_crowdsap=pre_cfg["min_crowdsap"],
                    max_rms_threshold=pre_cfg["max_rms_threshold"],
                )
            else:
                return []

        if not qf_result.passed:
            return []

        # ── Detrending ──────────────────────────────────────────────────
        detrend_result = detrend_light_curve(
            qf_result.time,
            qf_result.flux,
            qf_result.flux_err,
            sigma_clip_threshold=pre_cfg["sigma_clip_threshold"],
            detrend_method=pre_cfg["detrend_method"],
            detrend_window_days=pre_cfg["detrend_window_days"],
            gap_threshold_days=pre_cfg.get("gap_threshold_days", 0.5),
        )

        if len(detrend_result.time) < pre_cfg["min_datapoints"]:
            return []

        # ── Detection ───────────────────────────────────────────────────
        tces = search_multi_planet(
            detrend_result.time,
            detrend_result.flux,
            detrend_result.flux_err,
            tic_id=tic_id,
            r_star=r_star,
            m_star=m_star,
            teff=teff,
            period_min=det_cfg["period_min_days"],
            period_max=det_cfg["period_max_days"],
            snr_threshold=det_cfg["snr_threshold"],
            snr_discard_threshold=det_cfg.get("snr_discard_threshold", 5.0),
            n_trial_periods=det_cfg["n_trial_periods"],
            max_planets=det_cfg["max_planets_per_star"],
        )

        if not tces:
            return []

        # ── Per-TCE Processing: Phase Fold → Vetting → Features ────────
        candidates: list[dict] = []

        for tce in tces:
            if tce.snr_category == "noise":
                continue

            # Phase fold
            pf = phase_fold(
                detrend_result.time,
                detrend_result.flux,
                detrend_result.flux_err,
                period=tce.period,
                t0=tce.t0,
                duration=tce.duration,
            )

            # Only vet if SNR ≥ 7 (not marginal)
            if tce.snr_category in ("standard", "high"):
                # Run vetting
                vetting_result = run_all_vetting(
                    time=detrend_result.time,
                    flux=detrend_result.flux,
                    phase=pf.phase,
                    phase_flux=pf.flux,
                    binned_phase=pf.binned_phase,
                    binned_flux=pf.binned_flux,
                    transit_times=tce.transit_times,
                    period=tce.period,
                    t0=tce.t0,
                    duration_days=tce.duration,
                    depth=tce.depth,
                    r_star=r_star,
                    teff=teff,
                    tpf=None,  # TPF loading handled separately
                    config=vet_cfg,
                )

                # Extract features
                features = extract_features(
                    phase=pf.phase,
                    flux=pf.flux,
                    flux_err=pf.flux_err,
                    binned_phase=pf.binned_phase,
                    binned_flux=pf.binned_flux,
                    period=tce.period,
                    t0=tce.t0,
                    duration_days=tce.duration,
                    depth=tce.depth,
                    snr=tce.snr,
                    n_transit=tce.n_transits,
                    r_star=r_star,
                    teff=teff,
                    crowdsap=crowdsap,
                    transit_times=tce.transit_times,
                    time=detrend_result.time,
                    raw_flux=detrend_result.flux,
                    odd_even_sigma_value=vetting_result.odd_even_sigma,
                    secondary_depth_ratio_value=vetting_result.secondary_depth_ratio,
                    centroid_shift_value=vetting_result.centroid_shift,
                    flat_bottom_ratio_value=vetting_result.flat_bottom_ratio,
                    ingress_symmetry_value=vetting_result.ingress_symmetry,
                    duration_ratio_value=vetting_result.duration_ratio,
                )
            else:
                vetting_result = None
                features = {}

            # CNN input
            cnn_input = resample_phase_fold(pf.phase, pf.flux, target_length=256)

            # Build candidate record
            candidate = {
                **tce.to_dict(),
                "TIC_ID": tic_id,
                "r_star": r_star,
                "t_eff": teff,
                "crowdsap": crowdsap,
                "stellar_activity_flag": qf_result.stellar_activity_flag,
                "vetting_passed": vetting_result.vetting_passed if vetting_result else True,
                "hard_rejection_cause": vetting_result.rejection_cause if vetting_result else "NONE",
                "batman_fit": False,
                "multiplicity_boost_applied": False,
                "notes": "",
                "r_planet_earth_radii": features.get("r_planet_earth", np.nan) if features else np.nan,
                "_features": features,
                "_feature_array": features_to_array(features) if features else np.full(19, np.nan),
                "_cnn_input": cnn_input,
                "_vetting_results": (
                    vetting_result.test_results if vetting_result else {}
                ),
                "_time": detrend_result.time,
                "_flux": detrend_result.flux,
                "_trend": detrend_result.trend,
                "_phase": pf.phase,
                "_phase_flux": pf.flux,
                "_binned_phase": pf.binned_phase,
                "_binned_flux": pf.binned_flux,
                "_bls_periods": np.array([]),
                "_bls_powers": np.array([]),
            }

            # Hard-rejected candidates get their rejection cause as class label
            if vetting_result and vetting_result.hard_rejected:
                candidate["class_label"] = vetting_result.rejection_cause
                candidate["confidence_planet"] = 0.0
                candidate["confidence_eb"] = 1.0 if "ECLIPSING" in vetting_result.rejection_cause else 0.0
                candidate["confidence_blend"] = 1.0 if "BACKGROUND" in vetting_result.rejection_cause else 0.0
                candidate["confidence_other"] = 0.0
                candidate["tier"] = "false_positive"

            candidates.append(candidate)

        return candidates

    def _save_training_data(self, candidates: list[dict]) -> None:
        """Save feature vectors and CNN inputs to disk for ML training.

        Writes:
            data/candidates/features.csv — 19 features + TIC_ID + candidate_number.
            data/candidates/phase_folded_curves.npy — CNN input vectors.

        Args:
            candidates: List of candidate dictionaries from processing stage.
        """
        from astra.extraction.feature_extractor import FEATURE_NAMES

        if not candidates:
            logger.warning("No candidates to save for training")
            return

        candidates_dir = Path("data/candidates")
        candidates_dir.mkdir(parents=True, exist_ok=True)

        # Build feature rows
        feature_rows = []
        cnn_inputs = []

        for cand in candidates:
            features = cand.get("_features", {})
            if not features:
                continue

            row = {
                "TIC_ID": cand.get("TIC_ID", 0),
                "candidate_number": cand.get("candidate_number", 0),
            }
            for fname in FEATURE_NAMES:
                row[fname] = features.get(fname, np.nan)
            feature_rows.append(row)

            cnn_input = cand.get("_cnn_input")
            if cnn_input is not None:
                cnn_inputs.append(cnn_input)

        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            features_path = candidates_dir / "features.csv"
            features_df.to_csv(features_path, index=False)
            logger.info(
                f"Saved {len(features_df)} feature vectors to {features_path}"
            )

        if cnn_inputs:
            curves_array = np.array(cnn_inputs)
            curves_path = candidates_dir / "phase_folded_curves.npy"
            np.save(curves_path, curves_array)
            logger.info(
                f"Saved {len(cnn_inputs)} CNN input curves to {curves_path}"
            )

    def _run_classification(self, candidates: list[dict]) -> list[dict]:
        """Run ML classification on vetted candidates."""
        from astra.classification.multiplicity_boost import apply_multiplicity_boost
        from astra.extraction.feature_extractor import features_to_array

        if not candidates:
            return candidates

        # Load classifiers
        self._load_classifiers()

        if self._rf_classifier is None or self._rf_classifier.model is None:
            logger.warning("ML classifiers not available — skipping classification")
            return candidates

        for cand in candidates:
            # Skip already-classified (hard-rejected) candidates
            if cand.get("vetting_passed") is False:
                continue
            if cand.get("tier") == "marginal_unvetted":
                continue

            feature_array = cand.get("_feature_array", np.full(19, np.nan))

            # Phase 1: RF binary classification
            try:
                rf_labels, rf_proba = self._rf_classifier.predict(
                    feature_array.reshape(1, -1)
                )
                if rf_labels[0] == 0 and rf_proba[0] < 0.2:
                    # NOISE with >80% confidence → discard
                    cand["class_label"] = "NOISE"
                    cand["confidence_planet"] = 0.0
                    cand["tier"] = "false_positive"
                    continue
            except Exception as e:
                logger.warning(f"RF classification failed: {e}")

            # Phase 2: XGBoost multi-class
            try:
                if self._xgb_classifier and self._xgb_classifier.model:
                    xgb_labels, xgb_proba = self._xgb_classifier.predict(
                        feature_array.reshape(1, -1)
                    )
                    proba_xgb = xgb_proba[0]
                else:
                    proba_xgb = np.array([0.25, 0.25, 0.25, 0.25])
            except Exception as e:
                logger.warning(f"XGBoost classification failed: {e}")
                proba_xgb = np.array([0.25, 0.25, 0.25, 0.25])

            # CNN classification
            proba_cnn = None
            try:
                if self._cnn_classifier and self._cnn_classifier.model:
                    cnn_input = cand.get("_cnn_input")
                    if cnn_input is not None:
                        _, cnn_proba = self._cnn_classifier.predict(
                            cnn_input.reshape(1, -1)
                        )
                        proba_cnn = cnn_proba[0]
            except Exception as e:
                logger.warning(f"CNN classification failed: {e}")

            # Ensemble combination
            ensemble_result = self._ensemble.combine(proba_xgb, proba_cnn)

            # Assign final tier
            snr = cand.get("snr", 0)
            rep_cfg = self.config["reporting"]
            final_tier = self._ensemble.assign_final_tier(
                ensemble_result, snr,
                tier1_snr_min=rep_cfg.get("tier1_snr_min", 10.0),
                tier1_confidence_min=rep_cfg.get("tier1_confidence_min", 0.90),
                tier2_snr_min=rep_cfg.get("tier2_snr_min", 7.0),
                tier2_confidence_min=rep_cfg.get("tier2_confidence_min", 0.80),
            )

            cand["class_label"] = ensemble_result["class_label"]
            cand["confidence_planet"] = ensemble_result["confidence_planet"]
            cand["confidence_eb"] = ensemble_result["confidence_eb"]
            cand["confidence_blend"] = ensemble_result["confidence_blend"]
            cand["confidence_other"] = ensemble_result["confidence_other"]
            cand["tier"] = final_tier

        # Multiplicity boost
        if self.config["classification"].get("multiplicity_boost_enabled", True):
            candidates = apply_multiplicity_boost(
                candidates,
                max_boost_cap=self.config["classification"].get("multiplicity_boost_cap", 0.95),
            )

        return candidates

    def _run_reporting(
        self,
        candidates: list[dict],
        result: PipelineResult,
        sector: int,
        n_quality: int,
    ) -> PipelineResult:
        """Run the reporting stage."""
        from astra.reporting.catalogue_writer import write_catalogues, generate_statistics
        from astra.reporting.visualizer import generate_candidate_plot
        from astra.reporting.report_generator import generate_report

        output_dir = self.config["reporting"]["output_dir"]

        # Clean internal fields before writing catalogue
        clean_candidates = []
        for cand in candidates:
            clean = {k: v for k, v in cand.items() if not k.startswith("_")}
            clean_candidates.append(clean)

        # Write catalogues
        cat_paths = write_catalogues(
            clean_candidates,
            output_dir=f"{output_dir}/catalogues/",
            sector=sector,
        )
        result.full_catalogue_path = cat_paths.get("full", "")

        # Generate statistics
        stats = generate_statistics(
            clean_candidates,
            n_stars_processed=result.n_stars_processed,
            n_stars_passed_quality=n_quality,
            output_dir=f"{output_dir}/statistics/",
            sector=sector,
        )

        # Generate plots for Tier 1 and 2
        plot_tiers = self.config["reporting"].get("generate_plots_tiers", [1, 2])
        for cand in candidates:
            tier = cand.get("tier", "")
            tier_num = int(tier[-1]) if tier.startswith("tier") and tier[-1].isdigit() else 99
            if tier_num in plot_tiers:
                generate_candidate_plot(
                    tic_id=cand.get("TIC_ID", cand.get("tic_id", 0)),
                    candidate_number=cand.get("candidate_number", 0),
                    time=cand.get("_time", np.array([])),
                    flux=cand.get("_flux", np.array([])),
                    period=cand.get("period", 0),
                    t0=cand.get("t0", 0),
                    duration=cand.get("duration_days", 0),
                    phase=cand.get("_phase", np.array([])),
                    phase_flux=cand.get("_phase_flux", np.array([])),
                    binned_phase=cand.get("_binned_phase", np.array([])),
                    binned_flux=cand.get("_binned_flux", np.array([])),
                    vetting_results=cand.get("_vetting_results", {}),
                    classification_proba=np.array([
                        cand.get("confidence_planet", 0),
                        cand.get("confidence_eb", 0),
                        cand.get("confidence_blend", 0),
                        cand.get("confidence_other", 0),
                    ]),
                    output_dir=f"{output_dir}/plots/",
                    dpi=self.config["reporting"].get("plot_dpi", 300),
                )
        result.plot_directory = f"{output_dir}/plots/"

        # Generate PDF report
        top_candidates = sorted(
            clean_candidates,
            key=lambda c: c.get("confidence_planet", 0),
            reverse=True,
        )[:5]

        report_path = generate_report(
            sector=sector,
            statistics=stats,
            top_candidates=top_candidates,
            output_dir=f"{output_dir}/reports/",
        )
        result.report_path = report_path or ""

        # Count results
        result.n_planet_candidates = sum(
            1 for c in clean_candidates
            if c.get("tier") in ("tier1", "tier2", "tier3")
        )

        # Build tier DataFrames
        df = pd.DataFrame(clean_candidates) if clean_candidates else pd.DataFrame()
        if not df.empty and "tier" in df.columns:
            result.tier1_candidates = df[df["tier"] == "tier1"]
            result.tier2_candidates = df[df["tier"] == "tier2"]
            result.tier3_candidates = df[df["tier"] == "tier3"]

        return result
