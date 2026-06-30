# ASTRA — Automated Signal Transit Recognition Algorithm

> **Finding Worlds Hidden in Starlight**

**Team ONEROUS** — Bharatiya Antariksh Hackathon 2026 (BAH 2026), Challenge 7  
**Team Leader:** Raj Gupta  
**Members:** Saksham Gupta, Takshak Nikhil Khade, Dev Singi  
**Institution:** BITS Pilani Hyderabad Campus

---

## Overview

ASTRA is a complete, physics-informed AI pipeline for automated exoplanet transit detection and classification from TESS (Transiting Exoplanet Survey Satellite) light curve data. It processes stars per sector through a 6-stage sequential architecture:

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

---

## Installation

```bash
# Clone the repository
git clone https://github.com/DevS-ADG/ISRO-BAH.git
cd ISRO-BAH

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install the package and all dependencies
pip install -e .
# Or: pip install -r requirements.txt
```

### Dependencies
- Python ≥ 3.10
- lightkurve, astropy, wotan, transitleastsquares, batman-package
- scikit-learn, xgboost, PyTorch (CPU)
- astroquery, emcee, scipy, matplotlib, seaborn, reportlab

---

## ⚡ Quick Start with Pre-Trained Models

The trained RF and XGBoost models are included in `models/` — **no retraining needed**.

### Run the full pipeline on any TESS sector (models load automatically)
```bash
python scripts/run_pipeline.py --sector 1
```

The pipeline will automatically load models from `models/rf_phase1/` and `models/xgboost_phase2/` and classify all detected candidates.

### Classify your own feature vectors directly

```python
import joblib
import numpy as np
from astra.extraction.feature_extractor import FEATURE_NAMES

# Load the pre-trained models
rf  = joblib.load("models/rf_phase1/rf_model.joblib")
scaler_rf = joblib.load("models/rf_phase1/rf_scaler.joblib")

xgb = joblib.load("models/xgboost_phase2/xgb_model.joblib")
scaler_xgb = joblib.load("models/xgboost_phase2/xgb_scaler.joblib")

# Your 19-element feature vector (see FEATURE_NAMES for order)
# features = [period, depth, duration, snr, ...]
features = np.array([[...]])   # shape (n_samples, 19)

# Phase 1: Is it a signal or noise?
X_rf = scaler_rf.transform(features)
is_signal = rf.predict(X_rf)       # 0 = NOISE, 1 = SIGNAL
signal_prob = rf.predict_proba(X_rf)[:, 1]

# Phase 2: If signal, what class is it?
X_xgb = scaler_xgb.transform(features)
class_idx = xgb.predict(X_xgb)    # 0 = PLANET, 1 = ECLIPSING_BINARY
class_prob = xgb.predict_proba(X_xgb)
```

### Feature names (in order)
The 19 features expected by both models:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `period` | Best-fit orbital period (days) |
| 2 | `t0` | Transit epoch (BJD) |
| 3 | `duration` | Transit duration (hours) |
| 4 | `depth` | Transit depth (fractional flux) |
| 5 | `snr` | Signal-to-noise ratio |
| 6 | `bls_power` | BLS peak power |
| 7 | `tls_sde` | TLS Signal Detection Efficiency |
| 8 | `bls_fap` | BLS false alarm probability |
| 9 | `odd_even_diff` | Odd-even depth asymmetry |
| 10 | `secondary_sigma` | Secondary eclipse significance |
| 11 | `centroid_shift` | Centroid offset indicator |
| 12 | `oot_rms` | Out-of-transit scatter |
| 13 | `n_transits` | Number of observed transits |
| 14 | `transit_symmetry` | Transit shape symmetry score |
| 15 | `crowdsap` | Crowding metric (0–1) |
| 16 | `r_planet` | Derived planet radius (R_Earth) |
| 17 | `t_eq` | Equilibrium temperature (K) |
| 18 | `t_eff` | Stellar effective temperature (K) |
| 19 | `r_star` | Stellar radius (R_Sun) |

---

## Training on More Data

The included models were trained on **500 stars from TESS Sector 1** with labels cross-matched from ExoFOP TOI and the NASA Exoplanet Archive. To retrain on more sectors:

```bash
# 1. Run pipeline on more sectors (e.g. 2, 3, 4...)
python scripts/run_pipeline.py --sector 2

# 2. Regenerate labels (cross-matches TIC IDs with ExoFOP + NASA catalogs automatically)
python scripts/generate_labels.py

# 3. Retrain both models
python scripts/run_training.py --model rf
python scripts/run_training.py --model xgb
```

---

## Full Pipeline Usage

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

### Validate Labels
```bash
python scripts/validate_labels.py --labels data/labels/labels.csv
```

---

## Configuration

All tunable parameters are in `config/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.n_cores` | 4 | Parallel processing cores |
| `pipeline.max_stars` | 500 | Max stars per sector run |
| `detection.snr_threshold` | 7.0 | Minimum SNR for TCE |
| `detection.period_min_days` | 0.5 | Period search lower bound |
| `detection.period_max_days` | 13.0 | Period search upper bound |
| `classification.confidence_planet_threshold` | 0.80 | PLANET classification threshold |
| `batman_fitting.snr_threshold_for_fit` | 10.0 | Minimum SNR for BATMAN fitting |

---

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

---

## Project Structure

```
ISRO-BAH/
├── config/                  # Configuration files
├── data/
│   ├── candidates/          # features.csv, phase_folded_curves.npy
│   ├── labels/              # labels.csv (catalog cross-matched)
│   └── checkpoints/         # Pipeline resume checkpoints
├── models/
│   ├── rf_phase1/           # Trained Random Forest (binary)
│   └── xgboost_phase2/      # Trained XGBoost (multi-class)
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
│   ├── run_pipeline.py      # Main pipeline runner
│   ├── run_training.py      # Model training script
│   ├── generate_labels.py   # Catalog cross-matching for labels
│   └── validate_labels.py   # Label validation
├── tests/                   # Unit and integration tests
└── requirements.txt
```

---

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ -v --cov=astra
```

---

## License

MIT License — Team ONEROUS, BITS Pilani Hyderabad Campus


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
