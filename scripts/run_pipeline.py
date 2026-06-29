"""
ASTRA CLI Entry Point — Run the full pipeline or process single stars.

Usage:
    python run_pipeline.py --sector 1 --config config/config.yaml
    python run_pipeline.py --sector 1 --config config/config.yaml --resume
    python run_pipeline.py --tic_id 123456789 --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main CLI entry point for the ASTRA pipeline."""
    parser = argparse.ArgumentParser(
        description="ASTRA — Automated Signal Transit Recognition Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process full sector
    python run_pipeline.py --sector 1 --config config/config.yaml

    # Resume interrupted run
    python run_pipeline.py --sector 1 --config config/config.yaml --resume

    # Process single star
    python run_pipeline.py --tic_id 307210830 --sector 1 --config config/config.yaml
        """,
    )

    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        help="TESS sector number to process",
    )
    parser.add_argument(
        "--tic_id",
        type=int,
        default=None,
        help="TIC ID for single-star processing",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=None,
        help="Override number of parallel cores",
    )

    args = parser.parse_args()

    if args.tic_id is None and args.sector is None:
        parser.error("Either --sector or --tic_id must be specified")

    from astra.pipeline import ASTRAPipeline

    # Initialize pipeline
    pipeline = ASTRAPipeline(config_path=args.config)

    # Override n_cores if specified
    if args.n_cores is not None:
        pipeline.config["pipeline"]["n_cores"] = args.n_cores

    if args.tic_id is not None:
        # Single-star mode
        sector = args.sector or 1
        candidates = pipeline.run_single_star(args.tic_id, sector=sector)

        print(f"\n{'='*60}")
        print(f"  Results for TIC {args.tic_id}")
        print(f"{'='*60}")

        if candidates:
            for cand in candidates:
                print(
                    f"  Candidate {cand.get('candidate_number', '?')}: "
                    f"P={cand.get('period', 0):.4f}d, "
                    f"depth={cand.get('depth', 0):.5f}, "
                    f"SNR={cand.get('snr', 0):.1f}, "
                    f"class={cand.get('class_label', 'N/A')}, "
                    f"tier={cand.get('tier', 'N/A')}"
                )
        else:
            print("  No candidates detected")

        print(f"{'='*60}\n")

    else:
        # Full sector mode
        if args.resume:
            result = pipeline.resume(args.sector)
        else:
            result = pipeline.run(args.sector)

        print(f"\n{'='*60}")
        print(f"  ASTRA Pipeline — Sector {args.sector} Summary")
        print(f"{'='*60}")
        print(f"  Stars processed:      {result.n_stars_processed}")
        print(f"  TCEs found:           {result.n_tces}")
        print(f"  Planet candidates:    {result.n_planet_candidates}")
        print(f"  Tier 1 candidates:    {len(result.tier1_candidates)}")
        print(f"  Tier 2 candidates:    {len(result.tier2_candidates)}")
        print(f"  Tier 3 candidates:    {len(result.tier3_candidates)}")
        print(f"  Catalogue:            {result.full_catalogue_path}")
        print(f"  Report:               {result.report_path}")
        print(f"  Plots:                {result.plot_directory}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
