# ASTRA — Automated Signal Transit Recognition Algorithm

> **Finding Worlds Hidden in Starlight**

**Team ONEROUS** — Bharatiya Antariksh Hackathon 2026 (BAH 2026), Challenge 7  
**Team Leader:** Raj Gupta  
**Members:** Saksham Gupta, Takshak Nikhil Khade, Dev Singi  
**Institution:** BITS Pilani Hyderabad Campus

---

## Overview

ASTRA is a complete, physics-informed AI pipeline for automated exoplanet transit detection and classification from TESS (Transiting Exoplanet Survey Satellite) light curve data. It processes 20,000–30,000 stars per sector through a 6-stage sequential architecture:

1. **Signal Acquisition & Enhancement** — TESS FITS download, quality filtering, detrending
2. **Periodic Transit Search** — BLS + TLS dual search with iterative multi-planet detection
3. **Feature Extraction & Vetting** — 19-feature physics-informed vector, 6 astrophysical vetting tests
4. **ML Classification** — Random Forest (binary) → XGBoost + CNN ensemble (multi-class)
5. **Physical Model Fitting** — BATMAN (Mandel-Agol) with MCMC uncertainty estimation
6. **Reporting** — Tiered catalogue, 4-panel diagnostic figures, 3-page PDF report

## Architecture

```
DATA SOURCES → INGESTION → PREPROCESSING → DETECTION →
EXTRACTION & VETTING → CLASSIFICATION → ANALYSIS → OUTPUT
```

### ML Ensemble
- **Phase 1:** Random Forest (500 trees, balanced weights) — SIGNAL vs NOISE binary gate
- **Phase 2:** XGBoost (300 rounds, multi:softprob) — PLANET / EB / BLEND / OTHER
- **Phase 2:** 1D CNN (3-layer ConvNet, 128-dim embedding) — shape-based classification
- **Fusion:** P_final = 0.6 × P_xgb + 0.4 × P_cnn

### False Positive Rejection (3 independent layers)
1. **Astrophysical hard-rejection** (6 tests): odd-even depth, secondary eclipse, centroid shift, shape analysis, duration consistency, physical limits
2. **Feature-based classification** (19-parameter vector)
3. **Bayesian multiplicity boosting** for multi-planet systems

## Installation

```bash
# Clone and install
git clone <repository-url>
cd astra
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Dependencies
- Python ≥ 3.10
- lightkurve, astropy, wotan, transitleastsquares, batman-package
- scikit-learn, xgboost, PyTorch (CPU)
- emcee, scipy, matplotlib, seaborn, reportlab

## Usage

### Run Full Sector Pipeline
```bash
python scripts/run_pipeline.py --sector 1 --config config/config.yaml
```

### Resume Interrupted Run
```bash
python scripts/run_pipeline.py --sector 1 --config config/config.yaml --resume
```

### Process Single Star
```bash
python scripts/run_pipeline.py --tic_id 307210830 --sector 1 --config config/config.yaml
```

### Train ML Models
```bash
python scripts/run_training.py --config config/config.yaml --labels data/labels/labels.csv
```

### Validate Labels
```bash
python scripts/validate_labels.py --labels data/labels/labels.csv
```

## Configuration

All tunable parameters are in `config/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.n_cores` | 4 | Parallel processing cores |
| `detection.snr_threshold` | 7.0 | Minimum SNR for TCE |
| `detection.period_min_days` | 0.5 | Period search lower bound |
| `detection.period_max_days` | 13.0 | Period search upper bound |
| `classification.confidence_planet_threshold` | 0.80 | PLANET classification threshold |
| `batman_fitting.snr_threshold_for_fit` | 10.0 | Minimum SNR for BATMAN fitting |

## Output

### Candidate Catalogue (30 columns)
- `outputs/catalogues/candidates_full.csv` — All candidates
- `outputs/catalogues/candidates_tier1.csv` — Highest confidence
- `outputs/catalogues/candidates_tier2.csv` — Strong candidates
- `outputs/catalogues/candidates_tier3.csv` — Follow-up candidates

### Tiered Classification
| Tier | SNR | Confidence | Vetting | BATMAN |
|------|-----|------------|---------|--------|
| 1 | > 10 | > 0.90 | All passed | ✓ |
| 2 | > 7 | > 0.80 | All passed | Optional |
| 3 | > 7 | 0.50–0.80 | — | — |

### Visualizations
4-panel diagnostic figures per Tier 1/2 candidate at 300 DPI:
- Full detrended light curve with transit bands
- Phase-folded + BATMAN/BLS model overlay
- BLS/TLS periodogram with harmonics
- Vetting diagnostics + classification probabilities

### PDF Report
3-page structured report: Abstract & Methods → Results & Top Candidates → Discussion & Limitations

## Project Structure

```
astra/
├── config/                  # Configuration files
├── data/                    # Raw, processed, and label data
├── models/                  # Serialized ML models
├── outputs/                 # Catalogues, plots, reports
├── astra/                   # Main package
│   ├── ingestion/           # TESS data download & quality filter
│   ├── preprocessing/       # Detrending, normalization, gap handling
│   ├── detection/           # BLS, TLS, multi-planet search
│   ├── extraction/          # Phase fold, features, BATMAN fitting
│   ├── vetting/             # 6 astrophysical vetting tests
│   ├── classification/      # RF, XGBoost, CNN, ensemble
│   ├── reporting/           # Catalogues, plots, PDF reports
│   ├── training/            # Model training pipelines
│   ├── utils/               # Logging, checkpointing, parallel, stellar
│   └── pipeline.py          # Top-level orchestrator
├── scripts/                 # CLI entry points
├── tests/                   # Unit and integration tests
└── requirements.txt
```

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ -v --cov=astra
```

## License

MIT License — Team ONEROUS, BITS Pilani Hyderabad Campus
