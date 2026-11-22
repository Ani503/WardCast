"""
src/calibration_analysis.py
Step 2: Probability calibration evaluation for ANN and Bagging models.
Evaluates out-of-fold log-loss, multiclass Brier score, and reliability curves
under unweighted, class-weighted, and CalibratedClassifierCV regimes.
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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import load_raw_data, add_derived_features, get_preprocessor, STAY_BANDS

RANDOM_STATE = 42

def multiclass_brier_score(y_true, probas):
    """
    Computes multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    n_samples, n_classes = probas.shape
    y_one_hot = np.zeros((n_samples, n_classes))
    for i, y_val in enumerate(y_true):
        y_one_hot[i, int(y_val)] = 1.0
    return float(np.mean(np.sum((probas - y_one_hot) ** 2, axis=1)))

def plot_multiclass_reliability(y_true, probas, class_labels, title, filename):
    """
    Plots multiclass reliability curves (confidence vs observed accuracy).
    """
    n_classes = len(class_labels)
    plt.figure(figsize=(10, 7))
    
    # Also plot overall top-prediction calibration
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bins = np.linspace(0.0, 1.0, 11)
    bin_accs = []
    bin_confs = []
    
    for i in range(len(bins) - 1):
        idx = np.where((confidences >= bins[i]) & (confidences < bins[i+1]))[0]
        if len(idx) > 20:
            bin_accs.append(np.mean(accuracies[idx]))
            bin_confs.append(np.mean(confidences[idx]))
            
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    plt.plot(bin_confs, bin_accs, "s-", color="navy", lw=2, label="Top-Prediction Overall")
    
    # Class-specific curves
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_classes)))
    for c_idx in range(min(5, n_classes)):
        y_binary = (y_true == c_idx).astype(int)
        p_c = probas[:, c_idx]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_binary, p_c, n_bins=10, strategy="quantile")
        lbl = class_labels[c_idx] if c_idx < len(class_labels) else f"Class {c_idx}"
        plt.plot(mean_predicted_value, fraction_of_positives, "o--", alpha=0.7, label=f"Class: {lbl}")
        
    plt.xlabel("Mean Predicted Probability / Confidence", fontsize=11)
    plt.ylabel("Observed Empirical Accuracy / Frequency", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved calibration figure to {filename}")

def run_calibration_study():
    print("Loading Kaggle data for calibration analysis...")
    df = load_raw_data()
    df = add_derived_features(df)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    y_11 = df["Stay_label"].values
    y_3 = df["Stay_3class"].values
    groups = df["patientid"].values
    
    # Subsample to 50k patients for fast, rigorous calibration testing across multiple configs
    np.random.seed(RANDOM_STATE)
    unique_patients = np.unique(groups)
    selected_patients = set(np.random.choice(unique_patients, size=min(40000, len(unique_patients)), replace=False))
    sub_mask = df["patientid"].isin(selected_patients).values
    
    X_sub = X_df[sub_mask].reset_index(drop=True)
    y_11_sub = y_11[sub_mask]
    y_3_sub = y_3[sub_mask]
    groups_sub = groups[sub_mask]
    print(f"Calibration study cohort: {len(X_sub):,} records across {len(selected_patients):,} patient groups.")
    
    gkf = GroupKFold(n_splits=5)
    
    # Configurations to evaluate:
    # 1. ANN (Unweighted)
    # 2. ANN (Class-Weighted)
    # 3. Bagging (Unweighted)
    # 4. Bagging (Class-Weighted)
    # 5. Calibrated ANN (Sigmoid)
    
    targets = [
        {"name": "11-band", "y": y_11_sub, "n_classes": 11, "labels": STAY_BANDS},
        {"name": "3-class", "y": y_3_sub, "n_classes": 3, "labels": ["Short (<=20d)", "Medium (21-40d)", "Long (>40d)"]}
    ]
    
    calibration_metrics = {}
    
    for tgt in targets:
        tgt_name = tgt["name"]
        y_curr = tgt["y"]
        n_cls = tgt["n_classes"]
        labels = tgt["labels"]
        print(f"\n=======================================================")
        print(f"Evaluating Calibration on Target: {tgt_name} ({n_cls} classes)")
        print(f"=======================================================")
        
        models_to_test = {
            "ANN (Unweighted)": {"model_family": "ann", "weighted": False, "calibrated": False},
            "ANN (Class-Weighted)": {"model_family": "ann", "weighted": True, "calibrated": False},
            "Bagging (Unweighted)": {"model_family": "bagging", "weighted": False, "calibrated": False},
            "Bagging (Class-Weighted)": {"model_family": "bagging", "weighted": True, "calibrated": False},
            "ANN (Calibrated Sigmoid)": {"model_family": "ann", "weighted": False, "calibrated": "sigmoid"},
            "ANN (Calibrated Isotonic)": {"model_family": "ann", "weighted": False, "calibrated": "isotonic"},
        }
        
        tgt_results = {}
        
        for m_name, m_cfg in models_to_test.items():
            oof_probas = np.zeros((len(X_sub), n_cls))
            oof_preds = np.zeros(len(X_sub), dtype=int)
            
            for train_idx, val_idx in gkf.split(X_sub, y_curr, groups=groups_sub):
                X_tr, y_tr = X_sub.iloc[train_idx], y_curr[train_idx]
                X_va, y_va = X_sub.iloc[val_idx], y_curr[val_idx]
                
                prep = get_preprocessor(include_deposit=True, scale=True)
                X_tr_t = prep.fit_transform(X_tr)
                X_va_t = prep.transform(X_va)
                
                weights = compute_sample_weight("balanced", y_tr) if m_cfg["weighted"] else None
                
                if m_cfg["model_family"] == "ann":
                    base_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=RANDOM_STATE, early_stopping=True)
                    if m_cfg["calibrated"]:
                        # CalibratedClassifierCV
                        cal_clf = CalibratedClassifierCV(estimator=base_model, method=m_cfg["calibrated"], cv=3)
                        cal_clf.fit(X_tr_t, y_tr)
                        p_val = cal_clf.predict_proba(X_va_t)
                    else:
                        base_model.fit(X_tr_t, y_tr)
                        p_val = base_model.predict_proba(X_va_t)
                elif m_cfg["model_family"] == "bagging":
                    cw = "balanced" if m_cfg["weighted"] else None
                    base_dt = DecisionTreeClassifier(max_depth=10, class_weight=cw, random_state=RANDOM_STATE)
                    bag = BaggingClassifier(estimator=base_dt, n_estimators=15, random_state=RANDOM_STATE, n_jobs=-1)
                    bag.fit(X_tr_t, y_tr)
                    p_val = bag.predict_proba(X_va_t)
                    
                oof_probas[val_idx] = p_val
                oof_preds[val_idx] = np.argmax(p_val, axis=1)
                
            loss = float(log_loss(y_curr, oof_probas))
            brier = float(multiclass_brier_score(y_curr, oof_probas))
            acc = float(accuracy_score(y_curr, oof_preds))
            f1 = float(f1_score(y_curr, oof_preds, average="macro"))
            
            print(f"  {m_name:25s} | Log-Loss: {loss:0.4f} | Brier: {brier:0.4f} | Acc: {acc*100:5.2f}% | Macro-F1: {f1:0.4f}")
            
            tgt_results[m_name] = {
                "log_loss": loss,
                "brier_score": brier,
                "accuracy": acc,
                "macro_f1": f1
            }
            
            # Save reliability plot for selected top candidates
            if m_name in ["ANN (Unweighted)", "ANN (Class-Weighted)", "Bagging (Unweighted)", "ANN (Calibrated Sigmoid)"]:
                safe_name = m_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                plot_filename = f"reports/figures/calibration_{tgt_name}_{safe_name}.png"
                plot_multiclass_reliability(
                    y_true=y_curr,
                    probas=oof_probas,
                    class_labels=labels,
                    title=f"Reliability Curve: {m_name} ({tgt_name})",
                    filename=plot_filename
                )
                
        calibration_metrics[tgt_name] = tgt_results
        
    # Determine the best calibrated model overall
    # The lowest log-loss and Brier score on the 11-band target
    best_model_name = min(calibration_metrics["11-band"].items(), key=lambda x: x[1]["brier_score"])[0]
    print(f"\n=> Best Calibrated Model for Simulator: {best_model_name}")
    print(f"   Log-Loss: {calibration_metrics['11-band'][best_model_name]['log_loss']:.4f}")
    print(f"   Brier Score: {calibration_metrics['11-band'][best_model_name]['brier_score']:.4f}")
    
    # Save to metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["calibration_study"] = calibration_metrics
    metrics["best_calibrated_probability_engine"] = best_model_name
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Updated {metrics_path} with calibration study results.")
    return calibration_metrics, best_model_name

if __name__ == "__main__":
    run_calibration_study()
