"""
src/compute_roc_auc.py
Compute out-of-fold ROC-AUC metrics for the ANN (MLPClassifier):
- One-vs-rest macro ROC-AUC for 11-band target
- One-vs-rest macro ROC-AUC for 3-class target
- Binary ROC-AUC for Long class (>50 days)
- Save ROC curve figure to reports/figures/roc_curve_long_stay.png
- Update reports/metrics.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prep import load_raw_data, add_derived_features, get_preprocessor, STAY_BANDS, BAND_TO_3CLASS

RANDOM_STATE = 42

def main():
    print("Loading raw Kaggle data...")
    t0 = time.time()
    df = load_raw_data()
    df = add_derived_features(df)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    y_11 = df["Stay_label"].values
    y_3 = df["Stay_3class"].values
    groups = df["patientid"].values
    n_samples = len(df)
    
    print(f"Dataset shape: {X_df.shape}, {len(np.unique(groups))} unique patient groups.")
    
    oof_cache_file = "Data/processed/ann_oof_probabilities.npy"
    if os.path.exists(oof_cache_file):
        print(f"Loading cached out-of-fold probabilities from {oof_cache_file}...")
        oof_probs_11 = np.load(oof_cache_file)
    else:
        print("Computing 5-fold out-of-fold probabilities for ANN (MLPClassifier 64x32)...")
        oof_probs_11 = np.zeros((n_samples, 11), dtype=np.float32)
        gkf = GroupKFold(n_splits=5)
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_df, y_11, groups=groups), 1):
            f_start = time.time()
            X_tr, y_tr = X_df.iloc[train_idx], y_11[train_idx]
            X_va = X_df.iloc[val_idx]
            
            prep = get_preprocessor(include_deposit=True, scale=True)
            X_tr_t = prep.fit_transform(X_tr)
            X_va_t = prep.transform(X_va)
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=25,
                alpha=0.01,
                early_stopping=True,
                random_state=RANDOM_STATE
            )
            mlp.fit(X_tr_t, y_tr)
            p_val = mlp.predict_proba(X_va_t)
            oof_probs_11[val_idx] = p_val
            print(f"  Fold {fold_idx}/5 completed in {time.time() - f_start:.1f}s.")
            
        np.save(oof_cache_file, oof_probs_11)
        print(f"Saved out-of-fold probabilities to {oof_cache_file}.")

    # Verify probability constraints
    row_sums = np.sum(oof_probs_11, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "FATAL: Probabilities do not sum to 1.0"
    
    # 1. 11-band macro ROC-AUC
    auc_11_macro = float(roc_auc_score(y_11, oof_probs_11, multi_class="ovr", average="macro"))
    # Per-band AUC
    auc_11_per_band = {}
    for i, band in enumerate(STAY_BANDS):
        y_bin = (y_11 == i).astype(int)
        auc_b = float(roc_auc_score(y_bin, oof_probs_11[:, i]))
        auc_11_per_band[band] = round(auc_b, 4)
        
    print(f"\n11-Band Target One-vs-Rest Macro ROC-AUC: {auc_11_macro:.4f}")
    
    # 2. 3-class target macro ROC-AUC
    p_short = oof_probs_11[:, 0] + oof_probs_11[:, 1]
    p_medium = oof_probs_11[:, 2] + oof_probs_11[:, 3] + oof_probs_11[:, 4]
    p_long = np.sum(oof_probs_11[:, 5:], axis=1)
    oof_probs_3 = np.column_stack([p_short, p_medium, p_long])
    
    auc_3_macro = float(roc_auc_score(y_3, oof_probs_3, multi_class="ovr", average="macro"))
    auc_short = float(roc_auc_score((y_3 == 0).astype(int), p_short))
    auc_medium = float(roc_auc_score((y_3 == 1).astype(int), p_medium))
    auc_long = float(roc_auc_score((y_3 == 2).astype(int), p_long))
    
    print(f"3-Class Target One-vs-Rest Macro ROC-AUC: {auc_3_macro:.4f}")
    print(f"  Short (<=20d) ROC-AUC: {auc_short:.4f}")
    print(f"  Medium (21-50d) ROC-AUC: {auc_medium:.4f}")
    print(f"  Long (>50d) ROC-AUC: {auc_long:.4f}")
    
    # 3. Long class (>50 days) ROC curve
    y_long_binary = (y_3 == 2).astype(int)
    fpr, tpr, thresholds = roc_curve(y_long_binary, p_long)
    
    # Find operating point at threshold ~0.25
    idx_25 = np.argmin(np.abs(thresholds - 0.25))
    fpr_25 = fpr[idx_25]
    tpr_25 = tpr[idx_25]
    
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/roc_curve_long_stay.png"
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ANN Long-Stay ROC (AUC = {auc_long:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.5000)")
    plt.scatter([fpr_25], [tpr_25], color="#d62728", s=90, zorder=5, label=f"Selected Threshold θ=0.25 (Recall={tpr_25*100:.1f}%, FPR={fpr_25*100:.1f}%)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("ROC Curve: Long-Stay Inpatient Identification (>50 Days)\nANN (MLPClassifier 64x32, 5-Fold GroupKFold)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved ROC figure to {fig_path}")
    
    # Update metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["ann_roc_auc"] = {
        "11_band_ovr_macro_roc_auc": round(auc_11_macro, 4),
        "11_band_per_band_roc_auc": auc_11_per_band,
        "3_class_ovr_macro_roc_auc": round(auc_3_macro, 4),
        "3_class_short_roc_auc": round(auc_short, 4),
        "3_class_medium_roc_auc": round(auc_medium, 4),
        "3_class_long_roc_auc": round(auc_long, 4),
        "long_stay_binary_roc_auc": round(auc_long, 4),
        "operating_threshold_0_25_recall": round(float(tpr_25), 4),
        "operating_threshold_0_25_fpr": round(float(fpr_25), 4),
        "evaluation_provenance": "Out-of-fold predicted probabilities across 5-fold GroupKFold(patientid) on 318,438 Kaggle hospital records (ANN MLPClassifier 64x32)."
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path} with ann_roc_auc metrics in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
