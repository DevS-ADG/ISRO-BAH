"""
ASTRA Catalogue Writer — Structured CSV output of candidate catalogues.

Produces the 30-column full CSV catalogue and filtered tier-specific catalogues.
"""

import pandas as pd
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.reporting.catalogue_writer")

# Full catalogue columns (30 columns as specified)
CATALOGUE_COLUMNS = [
    "TIC_ID",
    "candidate_number",
    "tier",
    "period_days",
    "period_err",
    "T0_btjd",
    "T0_err",
    "duration_hours",
    "duration_err",
    "depth_ppm",
    "depth_err",
    "snr",
    "sde",
    "n_transits",
    "class_label",
    "confidence_planet",
    "confidence_eb",
    "confidence_blend",
    "confidence_other",
    "Rp_Rs",
    "Rp_Rs_err",
    "a_Rs",
    "a_Rs_err",
    "impact_param",
    "r_planet_earth",
    "r_planet_err",
    "orbital_distance_AU",
    "t_eq_kelvin",
    "odd_even_sigma",
    "secondary_sigma",
    "centroid_shift_arcsec",
    "duration_ratio",
    "flat_bottom_ratio",
    "crowdsap",
    "stellar_activity_flag",
    "batman_fit",
    "vetting_passed",
    "hard_rejection_cause",
    "multiplicity_boost_applied",
    "notes",
]


def write_catalogues(
    candidates: list[dict],
    output_dir: str = "outputs/catalogues/",
    sector: int | None = None,
) -> dict[str, str]:
    """Write full and tier-filtered candidate catalogues.

    Produces:
    - candidates_full.csv: All candidates with 30+ columns
    - candidates_tier1.csv: SNR>10, confidence>0.90, vetting passed, BATMAN fitted
    - candidates_tier2.csv: SNR>7, confidence>0.80, vetting passed
    - candidates_tier3.csv: SNR>7, confidence 0.50-0.80, follow-up flagged

    Args:
        candidates: List of candidate dictionaries.
        output_dir: Output directory for CSV files.
        sector: TESS sector number for filename prefix.

    Returns:
        Dictionary mapping catalogue name to file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"sector_{sector}_" if sector else ""

    if not candidates:
        logger.warning("No candidates to write to catalogue")
        return {}

    # Build DataFrame
    df = pd.DataFrame(candidates)

    # Ensure all expected columns exist
    for col in CATALOGUE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Sort by tier priority and SNR
    tier_order = {"tier1": 0, "tier2": 1, "tier3": 2, "false_positive": 3}
    df["_tier_order"] = df["tier"].map(tier_order).fillna(4)
    df = df.sort_values(["_tier_order", "snr"], ascending=[True, False])
    df = df.drop(columns=["_tier_order"])

    # Write full catalogue
    full_path = out_dir / f"{prefix}candidates_full.csv"
    df.to_csv(full_path, index=False, float_format="%.6f")
    logger.info(f"Full catalogue: {len(df)} candidates → {full_path}")

    paths = {"full": str(full_path)}

    # Tier 1: SNR > 10, confidence > 0.90, vetting passed, BATMAN fitted
    tier1 = df[
        (df["tier"] == "tier1")
        | (
            (df["snr"] >= 10.0)
            & (df["confidence_planet"] >= 0.90)
            & (df["vetting_passed"] == True)
            & (df["batman_fit"] == True)
        )
    ]
    if len(tier1) > 0:
        tier1_path = out_dir / f"{prefix}candidates_tier1.csv"
        tier1.to_csv(tier1_path, index=False, float_format="%.6f")
        paths["tier1"] = str(tier1_path)
        logger.info(f"Tier 1 catalogue: {len(tier1)} candidates → {tier1_path}")

    # Tier 2: SNR > 7, confidence > 0.80, vetting passed
    tier2 = df[
        (df["tier"] == "tier2")
        | (
            (df["snr"] >= 7.0)
            & (df["confidence_planet"] >= 0.80)
            & (df["vetting_passed"] == True)
            & (~df.index.isin(tier1.index))
        )
    ]
    if len(tier2) > 0:
        tier2_path = out_dir / f"{prefix}candidates_tier2.csv"
        tier2.to_csv(tier2_path, index=False, float_format="%.6f")
        paths["tier2"] = str(tier2_path)
        logger.info(f"Tier 2 catalogue: {len(tier2)} candidates → {tier2_path}")

    # Tier 3: SNR > 7, confidence 0.50-0.80
    tier3 = df[
        (df["tier"] == "tier3")
        | (
            (df["snr"] >= 7.0)
            & (df["confidence_planet"] >= 0.50)
            & (df["confidence_planet"] < 0.80)
        )
    ]
    if len(tier3) > 0:
        tier3_path = out_dir / f"{prefix}candidates_tier3.csv"
        tier3.to_csv(tier3_path, index=False, float_format="%.6f")
        paths["tier3"] = str(tier3_path)
        logger.info(f"Tier 3 catalogue: {len(tier3)} candidates → {tier3_path}")

    return paths


def generate_statistics(
    candidates: list[dict],
    n_stars_processed: int,
    n_stars_passed_quality: int,
    output_dir: str = "outputs/statistics/",
    sector: int | None = None,
) -> dict:
    """Generate sector-level summary statistics.

    Args:
        candidates: List of all candidate dictionaries.
        n_stars_processed: Total stars processed.
        n_stars_passed_quality: Stars passing quality filter.
        output_dir: Output directory.
        sector: TESS sector number.

    Returns:
        Statistics dictionary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    stats = {
        "sector": sector,
        "total_stars_processed": n_stars_processed,
        "stars_passed_quality": n_stars_passed_quality,
        "total_tces": len(df),
        "hard_rejected": int(df["vetting_passed"].eq(False).sum()) if "vetting_passed" in df else 0,
        "classified_by_ml": int(df["vetting_passed"].eq(True).sum()) if "vetting_passed" in df else 0,
        "tier1_count": int((df["tier"] == "tier1").sum()) if "tier" in df else 0,
        "tier2_count": int((df["tier"] == "tier2").sum()) if "tier" in df else 0,
        "tier3_count": int((df["tier"] == "tier3").sum()) if "tier" in df else 0,
        "false_positives": int((df["tier"] == "false_positive").sum()) if "tier" in df else 0,
    }

    # Save as JSON
    import json

    prefix = f"sector_{sector}_" if sector else ""
    stats_path = out_dir / f"{prefix}summary_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_path}")

    return stats
