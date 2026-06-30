"""
ASTRA POC Model Training — Generate physically consistent synthetic data and train calibrated models.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra.classification.rf_classifier import RFClassifier
from astra.classification.xgb_classifier import XGBMultiClassifier
from astra.extraction.feature_extractor import FEATURE_NAMES

def generate_synthetic_samples(base_df: pd.DataFrame, num_samples: int = 400) -> pd.DataFrame:
    """Generate physically plausible synthetic candidates."""
    synthetic_rows = []
    
    # We will generate a balanced dataset of 4 classes: PLANET, ECLIPSING_BINARY, BLEND, NOISE
    
    classes = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]
    samples_per_class = num_samples // len(classes)
    
    # Use real base rows to keep stellar params consistent (e.g. T_eff, R_star)
    base_rows = base_df.to_dict('records')
    
    for cls in classes:
        for _ in range(samples_per_class):
            base = np.random.choice(base_rows).copy()
            
            # Base stellar parameters
            r_star = base.get("r_star", 1.0)
            if np.isnan(r_star) or r_star <= 0: r_star = 1.0
            
            # Perturb period and compute duration physically
            period = np.random.uniform(0.5, 20.0)
            base["period"] = period
            
            # Realistic transit depth
            if cls == "PLANET":
                depth = np.random.uniform(0.0001, 0.02)
                base["snr"] = np.random.uniform(7.5, 100.0)
                base["odd_even_sigma"] = np.random.uniform(0.0, 1.5)
                base["secondary_depth_ratio"] = np.random.uniform(0.0, 0.1)
                base["centroid_shift"] = np.random.uniform(0.0, 2.0)
                base["flat_bottom_ratio"] = np.random.uniform(0.4, 0.9)
            elif cls == "ECLIPSING_BINARY":
                depth = np.random.uniform(0.01, 0.2)
                base["snr"] = np.random.uniform(20.0, 500.0)
                base["odd_even_sigma"] = np.random.uniform(1.0, 20.0) # EBs often have alternating depths
                base["secondary_depth_ratio"] = np.random.uniform(0.2, 1.0)
                base["centroid_shift"] = np.random.uniform(0.0, 2.0)
                base["flat_bottom_ratio"] = np.random.uniform(0.0, 0.4) # V-shaped
            elif cls == "BLEND":
                depth = np.random.uniform(0.001, 0.05)
                base["snr"] = np.random.uniform(5.0, 50.0)
                base["odd_even_sigma"] = np.random.uniform(0.0, 2.0)
                base["secondary_depth_ratio"] = np.random.uniform(0.0, 0.2)
                base["centroid_shift"] = np.random.uniform(3.0, 15.0) # High centroid shift
                base["flat_bottom_ratio"] = np.random.uniform(0.2, 0.6)
            else: # NOISE
                depth = np.random.uniform(0.00001, 0.001)
                base["snr"] = np.random.uniform(2.0, 6.5) # Below 7.0 SNR
                base["odd_even_sigma"] = np.random.uniform(0.0, 5.0)
                base["secondary_depth_ratio"] = np.random.uniform(0.0, 5.0)
                base["centroid_shift"] = np.random.uniform(0.0, 10.0)
                base["flat_bottom_ratio"] = np.random.uniform(0.0, 1.0)
                
            base["depth"] = depth
            base["r_planet_earth"] = np.sqrt(depth) * r_star * 109.076 # Physical consistency
            
            # Duration proportional to P^(1/3)
            base["duration"] = 24.0 * (period / np.pi) * np.arcsin((1.0 / (period**(2/3) * 4.2)) * np.sqrt(max(0, (1+np.sqrt(depth))**2)))
            if np.isnan(base["duration"]): base["duration"] = 2.0
            base["duration_ratio"] = np.random.uniform(0.8, 1.2) # Close to expected for real objects
            if cls == "NOISE":
                base["duration_ratio"] = np.random.uniform(0.1, 5.0)
                
            base["label"] = cls
            synthetic_rows.append(base)
            
    return pd.DataFrame(synthetic_rows)


def main():
    print("=" * 60)
    print("ASTRA POC Model Training Script")
    print("=" * 60)
    
    features_path = Path("data/candidates/features.csv")
    if not features_path.exists():
        print("Features file not found. Generating dummy base...")
        base_df = pd.DataFrame([{"r_star": 1.0, "t_eff": 5778.0, "crowdsap": 1.0}])
    else:
        base_df = pd.read_csv(features_path)
        
    print("Generating physically plausible synthetic dataset...")
    df = generate_synthetic_samples(base_df, num_samples=600)
    
    # Ensure ALL 19 features exist
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0
            
    X = df[FEATURE_NAMES].values
    
    # ── Phase 1: RF Binary Classification (SIGNAL vs NOISE) ──
    print("\nTraining Calibrated Random Forest (Phase 1)...")
    y_bin = (df["label"] != "NOISE").astype(int).values
    
    rf = RFClassifier(n_estimators=100)
    rf.build()
    
    X_imputed = rf._impute_nan(X, fit=True)
    X_scaled = rf.scaler.fit_transform(X_imputed)
    
    # Calibrate RF
    calibrated_rf = CalibratedClassifierCV(rf.model, method='sigmoid', cv=3)
    calibrated_rf.fit(X_scaled, y_bin)
    rf.model = calibrated_rf
    
    y_pred_bin = rf.model.predict(X_scaled)
    print("RF Binary Accuracy:", np.mean(y_pred_bin == y_bin))
    print(classification_report(y_bin, y_pred_bin, target_names=["NOISE", "SIGNAL"]))
    rf.save()
    
    # ── Phase 2: XGBoost Multi-Class ──
    print("\nTraining Calibrated XGBoost (Phase 2)...")
    # Only train on SIGNALs
    signal_mask = y_bin == 1
    X_sig = X[signal_mask]
    y_multi_str = df["label"].values[signal_mask]
    
    from astra.classification.xgb_classifier import CLASS_TO_IDX
    y_multi = np.array([CLASS_TO_IDX.get(lbl, 3) for lbl in y_multi_str])
    
    xgb = XGBMultiClassifier(n_estimators=100)
    xgb.build()
    
    X_sig_imputed = xgb._impute_nan(X_sig, fit=True)
    X_sig_scaled = xgb.scaler.fit_transform(X_sig_imputed)
    
    xgb.model.set_params(objective="multi:softprob", num_class=4)
    # Calibrate XGB
    calibrated_xgb = CalibratedClassifierCV(xgb.model, method='sigmoid', cv=3)
    calibrated_xgb.fit(X_sig_scaled, y_multi)
    xgb.model = calibrated_xgb
    
    y_pred_multi = xgb.model.predict(X_sig_scaled)
    print("XGBoost Multi Accuracy:", np.mean(y_pred_multi == y_multi))
    print(classification_report(y_multi, y_pred_multi, target_names=["PLANET", "ECLIPSING_BINARY", "BLEND"], labels=[0, 1, 2]))
    xgb.save()
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_multi, y_pred_multi)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["PLANET", "EB", "BLEND", "OTHER"], yticklabels=["PLANET", "EB", "BLEND", "OTHER"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('POC XGBoost Confusion Matrix')
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "poc_confusion_matrix.png", dpi=300)
    plt.close()
    
    print("\nModel training complete! POC models saved successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
