"""
ASTRA Batch Runner — Run the full pipeline on multiple TIC IDs and print results in a table.

Usage:
    python scripts/run_batch.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# List of target stars to process
TARGET_TICS = [
    261136679,   # pi Mensae — confirmed multi-planet host (PLANET)
    29857954,    # Clean single transit at 9.47d          (PLANET)
    307210830,   # Multi-transit super-Earth system       (PLANET)
    100100827,   # Strong EB secondary eclipse flag       (ECLIPSING_BINARY)
    279741377,   # Quiet star — no transit signal         (NOISE)
]
SECTOR = None
CONFIG_PATH = "config/config.yaml"

# Column widths for the table
COL_WIDTHS = {
    "TIC_ID":       12,
    "Cand#":         6,
    "Period(d)":    10,
    "Depth":        10,
    "Duration(h)":  12,
    "SNR":           8,
    "R_p(R_E)":     10,
    "Class":        22,
    "P(planet)":    10,
    "Tier":         18,
    "Vetting":      10,
    "Rejection":    20,
}

HEADER_KEYS = list(COL_WIDTHS.keys())


def fmt(val, width):
    """Format a value to a fixed width string."""
    if isinstance(val, float):
        s = f"{val:.4f}"
    else:
        s = str(val)
    return s[:width].ljust(width)


def print_table_header():
    row = " | ".join(fmt(k, COL_WIDTHS[k]) for k in HEADER_KEYS)
    sep = "-+-".join("-" * COL_WIDTHS[k] for k in HEADER_KEYS)
    print(f"\n{row}")
    print(sep)


def print_table_row(tic_id, cand):
    rp_re = cand.get("r_planet_earth_radii", cand.get("r_planet", float("nan")))
    dur_h = cand.get("duration_days", 0) * 24.0
    values = {
        "TIC_ID":      tic_id,
        "Cand#":       cand.get("candidate_number", "?"),
        "Period(d)":   cand.get("period", float("nan")),
        "Depth":       cand.get("depth", float("nan")),
        "Duration(h)": dur_h,
        "SNR":         cand.get("snr", float("nan")),
        "R_p(R_E)":    rp_re,
        "Class":       cand.get("class_label", "N/A"),
        "P(planet)":   cand.get("confidence_planet", float("nan")),
        "Tier":        cand.get("tier", "N/A"),
        "Vetting":     "PASS" if cand.get("vetting_passed", True) else "FAIL",
        "Rejection":   cand.get("hard_rejection_cause", "NONE"),
    }
    row = " | ".join(fmt(values[k], COL_WIDTHS[k]) for k in HEADER_KEYS)
    print(row)


def print_no_candidates(tic_id):
    values = {
        "TIC_ID":      tic_id,
        "Cand#":       "--",
        "Period(d)":   "--",
        "Depth":       "--",
        "Duration(h)": "--",
        "SNR":         "--",
        "R_p(R_E)":    "--",
        "Class":       "NO CANDIDATES",
        "P(planet)":   "--",
        "Tier":        "--",
        "Vetting":     "--",
        "Rejection":   "--",
    }
    row = " | ".join(fmt(values[k], COL_WIDTHS[k]) for k in HEADER_KEYS)
    print(row)


def main():
    from astra.pipeline import ASTRAPipeline

    print("=" * 140)
    print("  ASTRA -- Batch Pipeline Run")
    print(f"  TIC IDs: {TARGET_TICS}")
    print(f"  Sector : {SECTOR}")
    print("=" * 140)

    pipeline = ASTRAPipeline(config_path=CONFIG_PATH)

    all_results = {}

    for tic_id in TARGET_TICS:
        print(f"\n[*] Processing TIC {tic_id} ...")
        try:
            candidates = pipeline.run_single_star(tic_id, sector=SECTOR)
            all_results[tic_id] = candidates
            n = len(candidates)
            print(f"    -> {n} candidate(s) found")
        except Exception as e:
            import traceback
            print(f"    [ERROR] TIC {tic_id} failed: {e}")
            traceback.print_exc()
            all_results[tic_id] = []

    # -- Print summary table ------------------------------------------------
    print("\n\n" + "=" * 140)
    print("  ASTRA BATCH RESULTS -- SUMMARY TABLE")
    print("=" * 140)

    print_table_header()

    for tic_id in TARGET_TICS:
        candidates = all_results.get(tic_id, [])
        if not candidates:
            print_no_candidates(tic_id)
        else:
            for cand in candidates:
                print_table_row(tic_id, cand)

    print("\n" + "=" * 140)

    # -- Also dump full JSON results ----------------------------------------
    output_path = Path("outputs/batch_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for tic_id, cands in all_results.items():
        clean = []
        for c in cands:
            row = {k: v for k, v in c.items() if not k.startswith("_")}
            for k, v in list(row.items()):
                try:
                    import numpy as np
                    if isinstance(v, (np.integer,)):
                        row[k] = int(v)
                    elif isinstance(v, (np.floating, float)) and not isinstance(v, bool):
                        row[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        row[k] = v.tolist()
                except Exception:
                    row[k] = str(v)
            clean.append(row)
        serializable[str(tic_id)] = clean

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Full results saved to: {output_path.resolve()}")
    print("=" * 140)


if __name__ == "__main__":
    main()
