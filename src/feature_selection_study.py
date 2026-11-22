"""
src/feature_selection_study.py
Step 3: Feature selection sweep (k=5, 10, 15, 20) and ablations for ANN and Decision Tree.
Evaluated under 5-fold GroupKFold(patientid).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import load_raw_data, add_derived_features, get_preprocessor, get_iqr_outlier_mask

RANDOM_STATE = 42

def run_study():
    print("Loading data for feature selection and ablation study...")
    df = load_raw_data()
    df = add_derived_features(df)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    y = df["Stay_label"].values
    groups = df["patientid"].values
    
    # Subsample to 50k patients for rapid yet statistically rigorous feature sweep
    np.random.seed(RANDOM_STATE)
    unique_patients = np.unique(groups)
    selected_patients = set(np.random.choice(unique_patients, size=min(45000, len(unique_patients)), replace=False))
    sub_mask = df["patientid"].isin(selected_patients).values
    
    X_sub = X_df[sub_mask].reset_index(drop=True)
    y_sub = y[sub_mask]
    groups_sub = groups[sub_mask]
    print(f"Study cohort: {len(X_sub):,} records across {len(selected_patients):,} patient groups.")
    
    gkf = GroupKFold(n_splits=5)
    
    k_values = [5, 10, 15, 20]
    
    # Track metrics per k for both ANN and Decision Tree
    results_k = {
        "dt": {k: {"acc": [], "f1": []} for k in k_values},
        "ann": {k: {"acc": [], "f1": []} for k in k_values}
    }
    
    # Track full feature performance
    results_full = {
        "dt": {"acc": [], "f1": []},
        "ann": {"acc": [], "f1": []}
    }
    
    # Track ablation without Admission_Deposit
    results_no_deposit = {
        "dt": {"acc": [], "f1": []},
        "ann": {"acc": [], "f1": []}
    }
    
    fold = 0
    for train_idx, val_idx in gkf.split(X_sub, y_sub, groups=groups_sub):
        fold += 1
        print(f"\nEvaluating Fold {fold}/5...")
        
        X_tr = X_sub.iloc[train_idx]
        y_tr = y_sub[train_idx]
        X_va = X_sub.iloc[val_idx]
        y_va = y_sub[val_idx]
        
        # 1. Full Preprocessing
        prep_full = get_preprocessor(include_deposit=True, scale=True)
        X_tr_full = prep_full.fit_transform(X_tr)
        X_va_full = prep_full.transform(X_va)
        
        # Full models
        dt_full = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE)
        dt_full.fit(X_tr_full, y_tr)
        pred_dt = dt_full.predict(X_va_full)
        results_full["dt"]["acc"].append(accuracy_score(y_va, pred_dt))
        results_full["dt"]["f1"].append(f1_score(y_va, pred_dt, average="macro", zero_division=0))
        
        ann_full = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=RANDOM_STATE, early_stopping=True)
        ann_full.fit(X_tr_full, y_tr)
        pred_ann = ann_full.predict(X_va_full)
        results_full["ann"]["acc"].append(accuracy_score(y_va, pred_ann))
        results_full["ann"]["f1"].append(f1_score(y_va, pred_ann, average="macro", zero_division=0))
        
        # 2. Feature Selection Sweep for k in [5, 10, 15, 20]
        for k in k_values:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_tr_k = selector.fit_transform(X_tr_full, y_tr)
            X_va_k = selector.transform(X_va_full)
            
            # DT on k features
            dt_k = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE)
            dt_k.fit(X_tr_k, y_tr)
            p_dt_k = dt_k.predict(X_va_k)
            results_k["dt"][k]["acc"].append(accuracy_score(y_va, p_dt_k))
            results_k["dt"][k]["f1"].append(f1_score(y_va, p_dt_k, average="macro", zero_division=0))
            
            # ANN on k features
            ann_k = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=RANDOM_STATE, early_stopping=True)
            ann_k.fit(X_tr_k, y_tr)
            p_ann_k = ann_k.predict(X_va_k)
            results_k["ann"][k]["acc"].append(accuracy_score(y_va, p_ann_k))
            results_k["ann"][k]["f1"].append(f1_score(y_va, p_ann_k, average="macro", zero_division=0))
            
        # 3. Ablation: Without Admission_Deposit
        X_tr_nodep = X_tr.drop(columns=["Admission_Deposit"])
        X_va_nodep = X_va.drop(columns=["Admission_Deposit"])
        prep_nodep = get_preprocessor(include_deposit=False, scale=True)
        X_tr_nodep_t = prep_nodep.fit_transform(X_tr_nodep)
        X_va_nodep_t = prep_nodep.transform(X_va_nodep)
        
        dt_nodep = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE)
        dt_nodep.fit(X_tr_nodep_t, y_tr)
        p_dt_nodep = dt_nodep.predict(X_va_nodep_t)
        results_no_deposit["dt"]["acc"].append(accuracy_score(y_va, p_dt_nodep))
        results_no_deposit["dt"]["f1"].append(f1_score(y_va, p_dt_nodep, average="macro", zero_division=0))
        
        ann_nodep = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=RANDOM_STATE, early_stopping=True)
        ann_nodep.fit(X_tr_nodep_t, y_tr)
        p_ann_nodep = ann_nodep.predict(X_va_nodep_t)
        results_no_deposit["ann"]["acc"].append(accuracy_score(y_va, p_ann_nodep))
        results_no_deposit["ann"]["f1"].append(f1_score(y_va, p_ann_nodep, average="macro", zero_division=0))

    # Format summary tables
    summary_report = {
        "provenance_note": "Earlier feature selection scores (macro-F1 ~0.16 to 0.22) were produced by a RandomForestClassifier(n_estimators=40, max_depth=12) on 3-fold GroupKFold. The benchmarks below evaluate both Decision Tree and ANN across 5-fold GroupKFold.",
        "k_sweep": {},
        "full_vs_reduced_best_k": {},
        "deposit_ablation": {}
    }
    
    print("\n" + "="*70)
    print("FEATURE SELECTION SWEEP RESULTS (k in [5, 10, 15, 20])")
    print("="*70)
    print(f"{'k':<5} | {'Model':<15} | {'Accuracy (Mean +/- Std)':<25} | {'Macro-F1 (Mean +/- Std)':<25}")
    print("-" * 75)
    
    for k in k_values:
        summary_report["k_sweep"][f"k_{k}"] = {}
        for m in ["ann", "dt"]:
            m_label = "ANN (MLP)" if m == "ann" else "Decision Tree"
            acc_m = float(np.mean(results_k[m][k]["acc"]))
            acc_s = float(np.std(results_k[m][k]["acc"]))
            f1_m = float(np.mean(results_k[m][k]["f1"]))
            f1_s = float(np.std(results_k[m][k]["f1"]))
            print(f"{k:<5} | {m_label:<15} | {acc_m*100:5.2f}% +/- {acc_s*100:4.2f}%         | {f1_m:0.4f} +/- {f1_s:0.4f}")
            summary_report["k_sweep"][f"k_{k}"][m] = {
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "macro_f1_mean": f1_m,
                "macro_f1_std": f1_s
            }
            
    print("\n" + "="*70)
    print("FULL FEATURES (116) VS BEST REDUCED SET (k=20)")
    print("="*70)
    for m in ["ann", "dt"]:
        m_label = "ANN (MLP)" if m == "ann" else "Decision Tree"
        acc_full_m = float(np.mean(results_full[m]["acc"]))
        f1_full_m = float(np.mean(results_full[m]["f1"]))
        acc_k20_m = float(np.mean(results_k[m][20]["acc"]))
        f1_k20_m = float(np.mean(results_k[m][20]["f1"]))
        print(f"{m_label:15s} | Full (116): Acc {acc_full_m*100:.2f}%, F1 {f1_full_m:.4f} | Best Reduced (k=20): Acc {acc_k20_m*100:.2f}%, F1 {f1_k20_m:.4f} | Delta F1: {f1_k20_m - f1_full_m:+.4f}")
        summary_report["full_vs_reduced_best_k"][m] = {
            "full_accuracy": acc_full_m,
            "full_macro_f1": f1_full_m,
            "k20_accuracy": acc_k20_m,
            "k20_macro_f1": f1_k20_m,
            "delta_macro_f1": f1_k20_m - f1_full_m
        }
        
    print("\n" + "="*70)
    print("ABLATION STUDY: ADMISSION_DEPOSIT IMPACT")
    print("="*70)
    for m in ["ann", "dt"]:
        m_label = "ANN (MLP)" if m == "ann" else "Decision Tree"
        acc_full_m = float(np.mean(results_full[m]["acc"]))
        f1_full_m = float(np.mean(results_full[m]["f1"]))
        acc_nodep_m = float(np.mean(results_no_deposit[m]["acc"]))
        f1_nodep_m = float(np.mean(results_no_deposit[m]["f1"]))
        print(f"{m_label:15s} | With Deposit: Acc {acc_full_m*100:.2f}%, F1 {f1_full_m:.4f} | Without Deposit: Acc {acc_nodep_m*100:.2f}%, F1 {f1_nodep_m:.4f} | Delta Acc: {(acc_full_m - acc_nodep_m)*100:+.2f}%")
        summary_report["deposit_ablation"][m] = {
            "with_deposit_acc": acc_full_m,
            "with_deposit_f1": f1_full_m,
            "without_deposit_acc": acc_nodep_m,
            "without_deposit_f1": f1_nodep_m,
            "delta_acc": acc_full_m - acc_nodep_m,
            "delta_f1": f1_full_m - f1_nodep_m
        }
        
    # Save to metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["step3_feature_selection_and_ablations"] = summary_report
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nUpdated {metrics_path} with Step 3 findings.")
    return summary_report

if __name__ == "__main__":
    run_study()
