"""
ASTRA Report Generator — 3-page PDF report using ReportLab.

Generates a structured scientific report:
  Page 1: Abstract, methods, tools, algorithmic choices
  Page 2: Classifier performance, sector results, top candidates
  Page 3: Discussion, limitations, future work
"""

from datetime import datetime
from pathlib import Path

from astra.utils.logger import get_logger

logger = get_logger("astra.reporting.report_generator")


def generate_report(
    sector: int,
    statistics: dict,
    top_candidates: list[dict],
    training_metrics: dict | None = None,
    output_dir: str = "outputs/reports/",
) -> str | None:
    """Generate the 3-page PDF report for a processed sector.

    Args:
        sector: TESS sector number.
        statistics: Sector-level summary statistics.
        top_candidates: List of top 3-5 candidate dictionaries.
        training_metrics: ML training evaluation metrics (optional).
        output_dir: Output directory for PDF files.

    Returns:
        Path to the generated PDF, or None on failure.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
    except ImportError:
        logger.error("reportlab not installed. Install with: pip install reportlab")
        return None

    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = out_dir / f"sector_{sector}_report.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "ASTRATitle", parent=styles["Title"],
            fontSize=18, spaceAfter=6 * mm,
        )
        heading_style = ParagraphStyle(
            "ASTRAHeading", parent=styles["Heading2"],
            fontSize=13, spaceAfter=3 * mm, spaceBefore=5 * mm,
        )
        body_style = ParagraphStyle(
            "ASTRABody", parent=styles["Normal"],
            fontSize=10, leading=14, spaceAfter=2 * mm,
        )
        small_style = ParagraphStyle(
            "ASTRASmall", parent=styles["Normal"],
            fontSize=9, leading=12, spaceAfter=1 * mm,
        )

        elements = []

        # ════════════════════════════════════════════════════════════════
        # PAGE 1: ABSTRACT AND METHODS
        # ════════════════════════════════════════════════════════════════

        elements.append(Paragraph(
            "ASTRA: Automated Signal Transit Recognition Algorithm",
            title_style,
        ))
        elements.append(Paragraph(
            f"Sector {sector} Analysis Report — {datetime.now().strftime('%Y-%m-%d')}",
            body_style,
        ))
        elements.append(Paragraph(
            "Team ONEROUS — Bharatiya Antariksh Hackathon 2026 (BAH 2026)",
            small_style,
        ))
        elements.append(Spacer(1, 5 * mm))

        # Abstract
        elements.append(Paragraph("Abstract", heading_style))

        n_stars = statistics.get("total_stars_processed", 0)
        n_tier1 = statistics.get("tier1_count", 0)
        n_tier2 = statistics.get("tier2_count", 0)
        n_tier3 = statistics.get("tier3_count", 0)
        total_candidates = n_tier1 + n_tier2 + n_tier3

        abstract = (
            f"We present ASTRA (Automated Signal Transit Recognition Algorithm), "
            f"a complete physics-informed AI pipeline for automated exoplanet transit "
            f"detection and classification from TESS light curve data. "
            f"Applied to TESS Sector {sector} ({n_stars} stars processed with "
            f"2-minute cadence data), the pipeline implements a 6-stage sequential "
            f"processing architecture combining Box Least Squares and Transit Least "
            f"Squares periodic searches, six independent astrophysical vetting tests, "
            f"and a hybrid ML ensemble (Random Forest + XGBoost + 1D CNN) for "
            f"candidate classification. "
            f"BATMAN (Mandel-Agol) physical transit model fitting with MCMC "
            f"uncertainty estimation validates high-confidence candidates. "
            f"The pipeline identified {total_candidates} planet candidates across "
            f"three confidence tiers ({n_tier1} Tier 1, {n_tier2} Tier 2, "
            f"{n_tier3} Tier 3), with a multi-layer false positive rejection "
            f"system achieving robust discrimination between planets, eclipsing "
            f"binaries, background blends, and instrumental artifacts."
        )
        elements.append(Paragraph(abstract, body_style))
        elements.append(Spacer(1, 3 * mm))

        # Pipeline Description
        elements.append(Paragraph("Pipeline Description", heading_style))

        pipeline_steps = [
            ("Signal Acquisition", "TESS FITS download via lightkurve with MAST archive access and quality flag filtering."),
            ("Preprocessing", "5σ sigma clipping, gap segmentation, wotan biweight detrending (0.75-day window), median normalization."),
            ("Detection", "Dual BLS + TLS search over 0.5–13.0 day period range, iterative multi-planet search with transit masking."),
            ("Feature Extraction", "19-parameter physics-informed feature vector, phase folding with optimal binning (duration/50)."),
            ("Astrophysical Vetting", "6 independent tests: odd-even depth, secondary eclipse, centroid shift, shape analysis, duration consistency, physical limits."),
            ("ML Classification", "3-model ensemble: Random Forest (binary), XGBoost (multi-class), 1D CNN (shape embedding), with multiplicity boost."),
            ("BATMAN Fitting", "Mandel-Agol physical model with differential evolution + emcee MCMC for Tier 1 candidates."),
            ("Reporting", "Tiered catalogue, 4-panel diagnostic figures, and this structured report."),
        ]

        for step_name, step_desc in pipeline_steps:
            elements.append(Paragraph(
                f"<b>{step_name}:</b> {step_desc}", small_style,
            ))
        elements.append(Spacer(1, 3 * mm))

        # Tools and Libraries
        elements.append(Paragraph("Tools and Libraries", heading_style))
        tools = (
            "lightkurve ≥2.4.0, astropy ≥5.3.0, wotan ≥1.10, "
            "transitleastsquares ≥1.0.31, batman-package ≥2.4.9, "
            "scikit-learn ≥1.3.0, xgboost ≥2.0.0, PyTorch ≥2.1.0, "
            "emcee ≥3.1.4, scipy ≥1.11.0, reportlab ≥4.0.0, "
            "matplotlib ≥3.7.0, seaborn ≥0.12.0"
        )
        elements.append(Paragraph(tools, small_style))
        elements.append(Spacer(1, 3 * mm))

        # Algorithmic Choices
        elements.append(Paragraph("Algorithmic Choices", heading_style))
        choices = [
            "TLS over raw BLS: Limb-darkened transit templates improve sensitivity for small planets compared to box-shaped BLS models.",
            "XGBoost over deep NN: Interpretable gradient-boosted trees provide physics-consistent classification without black-box opacity.",
            "19 physics-informed features: Each feature has direct astrophysical motivation; no proxy or arbitrary features are included.",
            "Biweight detrending: Robust to outliers with 0.75-day window that preserves transit signals while removing systematics.",
        ]
        for choice in choices:
            elements.append(Paragraph(f"• {choice}", small_style))

        elements.append(PageBreak())

        # ════════════════════════════════════════════════════════════════
        # PAGE 2: RESULTS
        # ════════════════════════════════════════════════════════════════

        elements.append(Paragraph("Results", heading_style))

        # Sector Results Table
        elements.append(Paragraph("Sector Processing Summary", heading_style))

        results_data = [
            ["Metric", "Value"],
            ["Total Stars Processed", str(n_stars)],
            ["Stars Passing Quality Filter", str(statistics.get("stars_passed_quality", "N/A"))],
            ["TCEs Above SNR > 7", str(statistics.get("total_tces", 0))],
            ["Hard-Rejected by Vetting", str(statistics.get("hard_rejected", 0))],
            ["Classified by ML", str(statistics.get("classified_by_ml", 0))],
            ["Tier 1 Candidates", str(n_tier1)],
            ["Tier 2 Candidates", str(n_tier2)],
            ["Tier 3 Candidates", str(n_tier3)],
            ["False Positives", str(statistics.get("false_positives", 0))],
        ]

        results_table = Table(results_data, colWidths=[120 * mm, 40 * mm])
        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ]))
        elements.append(results_table)
        elements.append(Spacer(1, 5 * mm))

        # Classifier Performance
        if training_metrics:
            elements.append(Paragraph("Classifier Performance", heading_style))

            if "test_auc" in training_metrics:
                elements.append(Paragraph(
                    f"Macro-averaged AUC: {training_metrics['test_auc']:.4f} "
                    f"(benchmark: Malik et al., AUC 0.948)",
                    body_style,
                ))

        # Top Candidates
        if top_candidates:
            elements.append(Paragraph("Top Candidates", heading_style))

            cand_header = [
                "TIC ID", "Period (d)", "Depth (ppm)", "SNR",
                "Rp (R⊕)", "T_eq (K)", "Confidence", "Tier"
            ]
            cand_data = [cand_header]

            for cand in top_candidates[:5]:
                cand_data.append([
                    str(cand.get("TIC_ID", "N/A")),
                    f"{cand.get('period_days', 0):.4f}",
                    f"{cand.get('depth_ppm', 0):.1f}",
                    f"{cand.get('snr', 0):.1f}",
                    f"{cand.get('r_planet_earth', 0):.2f}",
                    f"{cand.get('t_eq_kelvin', 0):.0f}",
                    f"{cand.get('confidence_planet', 0):.3f}",
                    str(cand.get("tier", "N/A")),
                ])

            cand_table = Table(cand_data)
            cand_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#d5f5e3")]),
            ]))
            elements.append(cand_table)

        elements.append(PageBreak())

        # ════════════════════════════════════════════════════════════════
        # PAGE 3: DISCUSSION AND LIMITATIONS
        # ════════════════════════════════════════════════════════════════

        elements.append(Paragraph("Discussion and Limitations", heading_style))

        # Known Limitations
        elements.append(Paragraph("Known Limitations", heading_style))
        limitations = [
            "<b>Period range:</b> Detection is limited to 0.5–13.0 days, excluding long-period planets beyond single-sector observability.",
            "<b>Active star false positives:</b> Stellar flares and rotational modulation can mimic transit signals; the stellar activity flag partially mitigates this but does not fully eliminate false positives from active stars.",
            "<b>No monotransit detection:</b> Single observed transit events from long-period planets are undetectable by the BLS/TLS iterative search.",
            "<b>Centroid analysis in crowded fields:</b> TPF centroid shift test is unreliable when neighboring star PSFs overlap significantly due to TESS's large pixel scale (21\"/pixel).",
            "<b>Training data limitations:</b> Model performance depends on the quality and completeness of the curated label set.",
        ]
        for lim in limitations:
            elements.append(Paragraph(f"• {lim}", small_style))
        elements.append(Spacer(1, 3 * mm))

        # Future Work
        elements.append(Paragraph("Future Work", heading_style))
        future = [
            "<b>Multi-sector analysis:</b> Stacking observations from multiple TESS sectors extends effective period sensitivity and improves SNR.",
            "<b>Transit Timing Variations:</b> TTV search for gravitational perturbations between planets in multi-planet systems.",
            "<b>Extended BATMAN fitting:</b> Apply physical model fitting to all Tier 2 candidates, not only Tier 1.",
            "<b>Automated follow-up prioritization:</b> Rank candidates by scientific interest metrics for ground-based telescope scheduling.",
        ]
        for fw in future:
            elements.append(Paragraph(f"• {fw}", small_style))
        elements.append(Spacer(1, 5 * mm))

        # Conclusion
        elements.append(Paragraph("Conclusion", heading_style))
        conclusion = (
            f"ASTRA demonstrates a complete, end-to-end transit detection pipeline "
            f"that processes {n_stars} stars from TESS Sector {sector} through "
            f"scientifically rigorous preprocessing, detection, vetting, and "
            f"classification stages. The multi-layer false positive rejection system "
            f"— combining 6 astrophysical vetting tests, a 3-model ML ensemble, "
            f"and Bayesian multiplicity boosting — ensures that candidates reaching "
            f"the final catalogue are physically motivated and scientifically credible. "
            f"BATMAN model fitting with MCMC uncertainties provides publication-quality "
            f"parameter estimates for the highest-confidence candidates. "
            f"The modular, configurable architecture enables rapid adaptation to "
            f"new TESS sectors and future survey data."
        )
        elements.append(Paragraph(conclusion, body_style))
        elements.append(Spacer(1, 5 * mm))

        # Footer
        elements.append(Paragraph(
            f"Report generated by ASTRA v1.0.0 — "
            f"Team ONEROUS — BITS Pilani Hyderabad Campus — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            small_style,
        ))

        # Build PDF
        doc.build(elements)
        logger.info(f"PDF report generated: {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"PDF report generation failed: {e}", exc_info=True)
        return None
