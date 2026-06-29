"""
ASTRA Visualizer — 4-panel diagnostic figures per candidate.

Generates one figure per Tier 1 and Tier 2 candidate at 300 DPI:
  Panel 1 (top-left):  Full detrended light curve with transit bands
  Panel 2 (top-right): Phase-folded + model overlay
  Panel 3 (bottom-left):  BLS/TLS periodogram with harmonics
  Panel 4 (bottom-right): Vetting diagnostics + classification probabilities
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.reporting.visualizer")


def generate_candidate_plot(
    tic_id: int,
    candidate_number: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    phase: np.ndarray,
    phase_flux: np.ndarray,
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    bls_periods: np.ndarray | None = None,
    bls_powers: np.ndarray | None = None,
    model_phase: np.ndarray | None = None,
    model_flux: np.ndarray | None = None,
    batman_fit: bool = False,
    vetting_results: dict | None = None,
    classification_proba: np.ndarray | None = None,
    output_dir: str = "outputs/plots/",
    dpi: int = 300,
) -> str | None:
    """Generate 4-panel diagnostic figure for a single candidate.

    Args:
        tic_id: TIC ID of the host star.
        candidate_number: Candidate number (1-indexed).
        time: Full detrended time array.
        flux: Full detrended flux array.
        period: Orbital period in days.
        t0: Transit mid-time.
        duration: Transit duration in days.
        phase: Unbinned phase-folded phase array.
        phase_flux: Unbinned phase-folded flux array.
        binned_phase: Binned phase array.
        binned_flux: Binned flux array.
        bls_periods: BLS/TLS period grid (for periodogram).
        bls_powers: BLS/TLS power values.
        model_phase: Model phase array (BATMAN or BLS box).
        model_flux: Model flux array.
        batman_fit: Whether BATMAN model was fitted.
        vetting_results: Dictionary of vetting test results.
        classification_proba: Probability vector [PLANET, EB, BLEND, OTHER].
        output_dir: Output directory for plot files.
        dpi: Figure resolution (default 300).

    Returns:
        Path to the saved figure, or None on failure.
    """
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"ASTRA Candidate: TIC {tic_id} — Candidate {candidate_number}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # ── Panel 1: Full Detrended Light Curve ────────────────────────
        ax1 = axes[0, 0]
        ax1.scatter(time, flux, s=0.5, c="gray", alpha=0.4, rasterized=True)

        # Highlight transit times with yellow bands
        if period > 0 and not np.isnan(t0):
            n_transits = max(1, int((time[-1] - time[0]) / period) + 2)
            for i in range(n_transits):
                t_mid = t0 + i * period
                if time[0] <= t_mid <= time[-1]:
                    ax1.axvspan(
                        t_mid - duration / 2,
                        t_mid + duration / 2,
                        color="gold",
                        alpha=0.3,
                        zorder=0,
                    )

        ax1.set_xlabel("Time (BTJD)", fontsize=11)
        ax1.set_ylabel("Normalized Flux", fontsize=11)
        ax1.set_title(f"Full Detrended Light Curve — TIC {tic_id}", fontsize=12)
        ax1.tick_params(labelsize=9)

        # ── Panel 2: Phase-Folded with Model Overlay ───────────────────
        ax2 = axes[0, 1]

        # Unbinned data as semi-transparent gray scatter
        ax2.scatter(
            phase, phase_flux, s=1, c="silver", alpha=0.3,
            label="Unbinned", rasterized=True,
        )

        # Binned data as larger teal circles
        ax2.scatter(
            binned_phase, binned_flux, s=20, c="teal",
            edgecolors="darkslategray", linewidths=0.5,
            label="Binned", zorder=3,
        )

        # Model overlay
        if model_flux is not None and model_phase is not None:
            if batman_fit:
                ax2.plot(
                    model_phase, model_flux, "r-", linewidth=2,
                    label="BATMAN fit", zorder=4,
                )
            else:
                ax2.plot(
                    model_phase, model_flux, "--", color="orange",
                    linewidth=2, label="BLS box model", zorder=4,
                )

        ax2.set_xlabel("Phase", fontsize=11)
        ax2.set_ylabel("Normalized Flux", fontsize=11)
        ax2.set_title(f"Phase-Folded — P = {period:.4f} days", fontsize=12)
        ax2.legend(fontsize=9, loc="lower right")
        ax2.set_xlim(-0.5, 0.5)
        ax2.tick_params(labelsize=9)

        # ── Panel 3: BLS/TLS Periodogram ───────────────────────────────
        ax3 = axes[1, 0]

        if bls_periods is not None and bls_powers is not None:
            ax3.plot(bls_periods, bls_powers, "k-", linewidth=0.5, alpha=0.8)

            # Mark detected peak
            ax3.axvline(
                period, color="red", linestyle="--", linewidth=1.5,
                label=f"P = {period:.4f} d",
            )

            # Mark harmonics
            for ratio, label in [(0.5, "P/2"), (2.0, "2P"), (1/3, "P/3"), (3.0, "3P")]:
                harmonic = period * ratio
                if bls_periods[0] <= harmonic <= bls_periods[-1]:
                    ax3.axvline(
                        harmonic, color="gray", linestyle=":",
                        linewidth=1, alpha=0.6,
                    )
                    ax3.text(
                        harmonic, np.max(bls_powers) * 0.95, label,
                        fontsize=8, ha="center", color="gray",
                    )

            ax3.set_xscale("log")
            ax3.legend(fontsize=9)

        ax3.set_xlabel("Period (days)", fontsize=11)
        ax3.set_ylabel("BLS/TLS Power", fontsize=11)
        ax3.set_title("BLS/TLS Periodogram", fontsize=12)
        ax3.tick_params(labelsize=9)

        # ── Panel 4: Vetting Diagnostics + Classification ──────────────
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Vetting results table
        if vetting_results:
            table_data = []
            colors = []

            test_names = ["odd_even", "secondary_eclipse", "centroid",
                         "shape", "duration", "physical_limits"]
            display_names = ["Odd-Even", "Secondary Eclipse", "Centroid Shift",
                            "Shape Analysis", "Duration Check", "Physical Limits"]

            for test_name, display_name in zip(test_names, display_names):
                if test_name in vetting_results:
                    result = vetting_results[test_name]
                    status = result.get("status", "N/A")
                    value = result.get("value", result.get("sigma", np.nan))
                    threshold = result.get("threshold", "")

                    if isinstance(value, float) and np.isfinite(value):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = "N/A"

                    if isinstance(threshold, float):
                        thresh_str = f"{threshold:.1f}"
                    else:
                        thresh_str = str(threshold)

                    table_data.append([display_name, value_str, thresh_str, status])

                    if status == "PASS":
                        colors.append(["white", "white", "white", "#d4edda"])
                    elif status in ("FAIL",):
                        colors.append(["white", "white", "white", "#f8d7da"])
                    elif status == "FLAG":
                        colors.append(["white", "white", "white", "#fff3cd"])
                    else:
                        colors.append(["white", "white", "white", "#e2e3e5"])

            if table_data:
                table = ax4.table(
                    cellText=table_data,
                    colLabels=["Test", "Value", "Threshold", "Status"],
                    cellColours=colors,
                    loc="upper center",
                    cellLoc="center",
                    bbox=[0.0, 0.45, 1.0, 0.55],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)

        # Classification probability bars
        if classification_proba is not None:
            class_names = ["PLANET", "EB", "BLEND", "OTHER"]
            bar_colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

            bar_ax = fig.add_axes([0.58, 0.08, 0.35, 0.15])
            bars = bar_ax.barh(
                class_names, classification_proba,
                color=bar_colors, edgecolor="black", linewidth=0.5,
            )

            for bar, prob in zip(bars, classification_proba):
                bar_ax.text(
                    bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.3f}", va="center", fontsize=9,
                )

            bar_ax.set_xlim(0, 1.15)
            bar_ax.set_xlabel("Probability", fontsize=10)
            bar_ax.set_title("Classification", fontsize=10, fontweight="bold")
            bar_ax.tick_params(labelsize=9)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Save
        plot_path = out_dir / f"TIC_{tic_id}_candidate_{candidate_number}.png"
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Plot saved: {plot_path}")
        return str(plot_path)

    except Exception as e:
        logger.error(f"Plot generation failed for TIC {tic_id}: {e}", exc_info=True)
        plt.close("all")
        return None
