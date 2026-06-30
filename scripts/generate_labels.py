"""
ASTRA Label Generator — Cross-match TIC IDs with public TESS catalogues.

Downloads the TESS Objects of Interest (TOI) catalogue from ExoFOP and
confirmed planet data from the NASA Exoplanet Archive. Cross-matches
TIC IDs from the extracted features to assign training labels.

Label mapping:
    PLANET           — Confirmed planets (TOI disposition CP/KP)
    ECLIPSING_BINARY — Known eclipsing binaries (TOI disposition EB)
    BLEND            — Background blends / instrumental (TOI disposition IS)
    OTHER            — False alarms or ambiguous (TOI disposition FA/FP/PC)
    NOISE            — No catalogue match (assumed noise)

Usage:
    python generate_labels.py --features data/candidates/features.csv
    python generate_labels.py --features data/candidates/features.csv --output data/labels/labels.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_toi_catalogue() -> "pd.DataFrame":
    """Download the TOI catalogue from ExoFOP.

    Returns:
        DataFrame with TOI dispositions indexed by TIC ID.
    """
    import pandas as pd

    # ExoFOP TOI table — publicly available CSV
    toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

    print("Downloading TOI catalogue from ExoFOP...")
    try:
        toi_df = pd.read_csv(toi_url)
        print(f"  Downloaded {len(toi_df)} TOI entries")
        return toi_df
    except Exception as e:
        print(f"  Warning: Could not download TOI catalogue: {e}")
        print("  Trying alternative NASA Exoplanet Archive source...")

        # Fallback: TAP query to NASA Exoplanet Archive
        try:
            alt_url = (
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
                "query=select+tid,toi,tfopwg_disp,rastr,decstr+"
                "from+toi&format=csv"
            )
            toi_df = pd.read_csv(alt_url)
            print(f"  Downloaded {len(toi_df)} TOI entries (alt source)")
            return toi_df
        except Exception as e2:
            print(f"  Error: Both TOI sources failed: {e2}")
            return pd.DataFrame()


def download_confirmed_planets() -> set:
    """Download confirmed TESS planets from NASA Exoplanet Archive.

    Returns:
        Set of TIC IDs with confirmed planets.
    """
    import pandas as pd

    print("Downloading confirmed planets from NASA Exoplanet Archive...")
    try:
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            "query=select+distinct+tic_id+from+ps+"
            "where+disc_facility+like+'%25TESS%25'+and+tic_id+is+not+null"
            "&format=csv"
        )
        df = pd.read_csv(url)
        # Handle 'TIC ' prefix
        clean_tics = df["tic_id"].dropna().astype(str).str.replace(r"^TIC\s+", "", regex=True)
        # Filter for purely numeric strings to avoid int() conversion errors
        clean_tics = clean_tics[clean_tics.str.isnumeric()]
        confirmed_tics = set(clean_tics.astype(int).values)
        print(f"  Found {len(confirmed_tics)} confirmed planet host TIC IDs")
        return confirmed_tics
    except Exception as e:
        print(f"  Warning: Could not download confirmed planets: {e}")
        return set()


def map_toi_disposition(disposition: str) -> str:
    """Map a TOI disposition string to an ASTRA label.

    Args:
        disposition: TFOPWG disposition string (e.g. 'KP', 'CP', 'EB', 'FA').

    Returns:
        ASTRA label string.
    """
    if not isinstance(disposition, str):
        return "OTHER"

    disposition = disposition.strip().upper()

    # Confirmed / Known planet
    if disposition in ("KP", "CP"):
        return "PLANET"

    # Eclipsing binary
    if disposition in ("EB",):
        return "ECLIPSING_BINARY"

    # Instrumental / background blend
    if disposition in ("IS", "V"):
        return "BLEND"

    # False alarm / false positive / planet candidate (ambiguous)
    if disposition in ("FA", "FP"):
        return "OTHER"

    # Planet candidate — treat as PLANET for training since it's the best guess
    if disposition in ("PC",):
        return "PLANET"

    return "OTHER"


def generate_labels(features_path: str, output_path: str) -> None:
    """Generate labels by cross-matching features with public catalogues.

    Args:
        features_path: Path to features.csv from pipeline.
        output_path: Path to write labels.csv.
    """
    import numpy as np
    import pandas as pd

    # Load features
    features_df = pd.read_csv(features_path)
    tic_ids = features_df["TIC_ID"].unique()
    print(f"\nLoaded {len(features_df)} candidates from {len(tic_ids)} unique stars")

    # Download catalogues
    toi_df = download_toi_catalogue()
    confirmed_tics = download_confirmed_planets()

    # Build TIC → label mapping
    tic_label_map = {}

    # Step 1: Map from TOI catalogue
    if not toi_df.empty:
        # Find the TIC ID column (varies between sources)
        tic_col = None
        for col_name in ["TIC ID", "TIC", "tid", "tic_id", "TIC_ID"]:
            if col_name in toi_df.columns:
                tic_col = col_name
                break

        disp_col = None
        for col_name in ["TFOPWG Disposition", "TFOPWG Disp", "tfopwg_disp", "TFOPWG_Disp", "Disposition"]:
            if col_name in toi_df.columns:
                disp_col = col_name
                break

        if tic_col and disp_col:
            for _, row in toi_df.iterrows():
                try:
                    tic = int(row[tic_col])
                    disp = str(row[disp_col])
                    label = map_toi_disposition(disp)
                    # Only overwrite if the new label is more specific
                    if tic not in tic_label_map or label == "PLANET":
                        tic_label_map[tic] = label
                except (ValueError, TypeError):
                    continue

            print(f"\n  TOI catalogue matched {len(tic_label_map)} TIC IDs")
        else:
            print(f"  Warning: Could not find TIC/disposition columns in TOI data")
            print(f"  Available columns: {list(toi_df.columns[:10])}")

    # Step 2: Override with confirmed planets
    for tic in confirmed_tics:
        tic_label_map[tic] = "PLANET"

    # Step 3: Build labels DataFrame
    label_rows = []
    n_matched = 0
    n_noise = 0

    for _, row in features_df.iterrows():
        tic_id = int(row["TIC_ID"])
        cand_num = int(row["candidate_number"])

        if tic_id in tic_label_map:
            label = tic_label_map[tic_id]
            n_matched += 1
        else:
            label = "NOISE"
            n_noise += 1

        label_rows.append({
            "TIC_ID": tic_id,
            "candidate_number": cand_num,
            "label": label,
        })

    labels_df = pd.DataFrame(label_rows)

    # Print distribution
    print(f"\n{'='*50}")
    print(f"  Label Distribution")
    print(f"{'='*50}")
    for label in sorted(labels_df["label"].unique()):
        count = (labels_df["label"] == label).sum()
        pct = 100 * count / len(labels_df)
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {len(labels_df):5d}")
    print(f"  Matched from catalogs: {n_matched}")
    print(f"  Unmatched (NOISE):     {n_noise}")
    print(f"{'='*50}")

    # Handle extreme imbalance: if ALL labels are NOISE, warn the user
    if n_matched == 0:
        print("\n  WARNING: No TIC IDs matched any catalogue entry.")
        print("  All candidates will be labelled NOISE.")
        print("  Training may not be meaningful without positive examples.")
        print("  Consider processing more stars or using a different sector.\n")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    print(f"\nLabels saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ASTRA Label Generator — Cross-match with TESS catalogues"
    )
    parser.add_argument(
        "--features", type=str, default="data/candidates/features.csv",
        help="Path to features CSV from pipeline",
    )
    parser.add_argument(
        "--output", type=str, default="data/labels/labels.csv",
        help="Path to output labels CSV",
    )
    args = parser.parse_args()

    if not Path(args.features).exists():
        print(f"Error: Features file not found: {args.features}")
        print("Run the pipeline first to generate features.")
        sys.exit(1)

    generate_labels(args.features, args.output)


if __name__ == "__main__":
    main()
