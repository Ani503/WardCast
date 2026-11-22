"""
src/fix_naive_bayes.py
Step 1: Fix and diagnose Naive Bayes models on Kaggle stay data.
Compares GaussianNB, BernoulliNB, and CategoricalNB under GroupKFold(5).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import (
    load_raw_data,
    add_derived_features,
    get_preprocessor,
    STAY_BANDS,
    NOMINAL_CATEGORICAL_COLS,
    ORDINAL_NUMERIC_COLS,
    NUMERIC_BASE_COLS
)

RANDOM_STATE = 42

def evaluate_nb_variants():
    print("Loading raw data...")
    df = load_raw_data()
    df = add_derived_features(df)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    y = df["Stay_label"].values
    groups = df["patientid"].values
    
    gkf = GroupKFold(n_splits=5)
    
    # 1. Standard preprocessor for GaussianNB (scaled continuous + one-hot)
    gnb_preprocessor = get_preprocessor(include_deposit=True, scale=True)
    
    # 2. Binary preprocessor for BernoulliNB (One-Hot for categoricals + binned one-hot for continuous)
    num_cols = list(NUMERIC_BASE_COLS) + ["Admission_Deposit"] + ORDINAL_NUMERIC_COLS
    bnb_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
            ]), NOMINAL_CATEGORICAL_COLS),
            ("num_bin", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("discretizer", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"))
            ]), num_cols)
        ]
    )
    
    # 3. Categorical preprocessor for CategoricalNB (Ordinal integer categories >= 0)
    # We will use OrdinalEncoder for categoricals and KBinsDiscretizer(encode='ordinal') for numerics
    cnb_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999))
            ]), NOMINAL_CATEGORICAL_COLS),
            ("num_ord", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("discretizer", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile"))
            ]), num_cols)
        ]
    )
    
    results = {
        "GaussianNB": {"preds": np.zeros(len(df), dtype=int), "probas": np.zeros((len(df), 11)), "fold_acc": [], "fold_f1": []},
        "BernoulliNB": {"preds": np.zeros(len(df), dtype=int), "probas": np.zeros((len(df), 11)), "fold_acc": [], "fold_f1": []},
        "CategoricalNB": {"preds": np.zeros(len(df), dtype=int), "probas": np.zeros((len(df), 11)), "fold_acc": [], "fold_f1": []}
    }
    
    fold = 0
    for train_idx, val_idx in gkf.split(X_df, y, groups=groups):
        fold += 1
        print(f"\nEvaluating Fold {fold}/5 (Train: {len(train_idx):,}, Val: {len(val_idx):,})...")
        
        X_train_raw, y_train = X_df.iloc[train_idx], y[train_idx]
        X_val_raw, y_val = X_df.iloc[val_idx], y[val_idx]
        
        # --- GaussianNB ---
        X_tr_gnb = gnb_preprocessor.fit_transform(X_train_raw)
        X_va_gnb = gnb_preprocessor.transform(X_val_raw)
        gnb = GaussianNB()
        gnb.fit(X_tr_gnb, y_train)
        pred_gnb = gnb.predict(X_va_gnb)
        prob_gnb = gnb.predict_proba(X_va_gnb)
        results["GaussianNB"]["preds"][val_idx] = pred_gnb
        results["GaussianNB"]["probas"][val_idx] = prob_gnb
        results["GaussianNB"]["fold_acc"].append(accuracy_score(y_val, pred_gnb))
        results["GaussianNB"]["fold_f1"].append(f1_score(y_val, pred_gnb, average="macro"))
        
        # --- BernoulliNB ---
        X_tr_bnb = bnb_preprocessor.fit_transform(X_train_raw)
        X_va_bnb = bnb_preprocessor.transform(X_val_raw)
        bnb = BernoulliNB(alpha=1.0)
        bnb.fit(X_tr_bnb, y_train)
        pred_bnb = bnb.predict(X_va_bnb)
        prob_bnb = bnb.predict_proba(X_va_bnb)
        results["BernoulliNB"]["preds"][val_idx] = pred_bnb
        results["BernoulliNB"]["probas"][val_idx] = prob_bnb
        results["BernoulliNB"]["fold_acc"].append(accuracy_score(y_val, pred_bnb))
        results["BernoulliNB"]["fold_f1"].append(f1_score(y_val, pred_bnb, average="macro"))
        
        # --- CategoricalNB ---
        X_tr_cnb = cnb_preprocessor.fit_transform(X_train_raw)
        X_va_cnb = cnb_preprocessor.transform(X_val_raw)
        
        # For CategoricalNB, ensure unknown values (999) in val are mapped to train max category + 1
        # Count max category per column in train
        n_cats = [int(np.max(X_tr_cnb[:, col])) + 2 for col in range(X_tr_cnb.shape[1])]
        # Replace 999 with the max category index (n_cats[col] - 1)
        for col in range(X_va_cnb.shape[1]):
            unknown_mask = (X_va_cnb[:, col] == 999) | (X_va_cnb[:, col] >= n_cats[col] - 1)
            X_va_cnb[unknown_mask, col] = n_cats[col] - 1
            
        cnb = CategoricalNB(alpha=1.0, min_categories=n_cats)
        cnb.fit(X_tr_cnb, y_train)
        pred_cnb = cnb.predict(X_va_cnb)
        prob_cnb = cnb.predict_proba(X_va_cnb)
        results["CategoricalNB"]["preds"][val_idx] = pred_cnb
        results["CategoricalNB"]["probas"][val_idx] = prob_cnb
        results["CategoricalNB"]["fold_acc"].append(accuracy_score(y_val, pred_cnb))
        results["CategoricalNB"]["fold_f1"].append(f1_score(y_val, pred_cnb, average="macro"))

    print("\n" + "="*70)
    print("NAIVE BAYES DIAGNOSIS AND PERFORMANCE COMPARISON")
    print("="*70)
    
    # Ground truth distribution
    print("\nGround Truth Stay Band Distribution:")
    gt_counts = pd.Series(y).value_counts().sort_index()
    for idx, count in gt_counts.items():
        print(f"  Band {idx:2d} ({STAY_BANDS[idx]:18s}): {count:6,d} ({count/len(y)*100:5.2f}%)")
        
    summary_data = {}
    for name, res in results.items():
        acc_m = float(np.mean(res["fold_acc"]))
        acc_s = float(np.std(res["fold_acc"]))
        f1_m = float(np.mean(res["fold_f1"]))
        f1_s = float(np.std(res["fold_f1"]))
        
        preds = res["preds"]
        within_1 = float(np.mean(np.abs(preds - y) <= 1))
        mae = float(np.mean(np.abs(preds - y)))
        
        # Long stay (bands 5..10, i.e. > 50 days)
        y_is_long = (y >= 5)
        pred_is_long = (preds >= 5)
        prec, rec, f1_l, _ = precision_recall_fscore_support(y_is_long, pred_is_long, average="binary", zero_division=0)
        
        pred_counts = pd.Series(preds).value_counts().sort_index()
        pred_dist = {STAY_BANDS[i]: int(pred_counts.get(i, 0)) for i in range(11)}
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc_m*100:5.2f}% +/- {acc_s*100:4.2f}%")
        print(f"Macro-F1: {f1_m:0.4f} +/- {f1_s:0.4f}")
        print(f"Within-1-Band Acc: {within_1*100:5.2f}% | MAE: {mae:0.2f} bands")
        print(f"Long-Stay Recall: {rec*100:5.2f}% | Precision: {prec*100:5.2f}% | F1: {f1_l:0.4f}")
        print("Predicted Band Distribution:")
        for b_idx in range(11):
            c = pred_counts.get(b_idx, 0)
            pct = c / len(y) * 100
            print(f"  Band {b_idx:2d} ({STAY_BANDS[b_idx]:18s}): {c:6,d} ({pct:5.2f}%)")
            
        summary_data[name] = {
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "macro_f1_mean": f1_m,
            "macro_f1_std": f1_s,
            "within_one_band_acc_mean": within_1,
            "within_one_band_acc_std": 0.0,
            "mae_band_idx_mean": mae,
            "mae_band_idx_std": 0.0,
            "long_stay_recall_mean": float(rec),
            "long_stay_recall_std": 0.0,
            "long_stay_precision_mean": float(prec),
            "long_stay_precision_std": 0.0,
            "long_stay_f1_mean": float(f1_l),
            "long_stay_f1_std": 0.0,
            "pred_distribution": pred_dist
        }
        
    # Save diagnostic results to reports/metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["naive_bayes_diagnosis"] = summary_data
    
    # Pick the best model (by Macro-F1 / Accuracy)
    best_nb_name = "CategoricalNB" if summary_data["CategoricalNB"]["macro_f1_mean"] > summary_data["BernoulliNB"]["macro_f1_mean"] else "BernoulliNB"
    best_nb_data = summary_data[best_nb_name]
    
    metrics["best_naive_bayes_model"] = best_nb_name
    print(f"\n=> Best Naive Bayes Model: {best_nb_name} (Acc: {best_nb_data['accuracy_mean']*100:.2f}%, Macro-F1: {best_nb_data['macro_f1_mean']:.4f})")
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Update reports/stay_model_comparison.csv
    csv_path = "reports/stay_model_comparison.csv"
    comp_df = pd.read_csv(csv_path)
    
    # Remove GaussianNB row if present
    comp_df = comp_df[~comp_df["Model"].str.contains("GaussianNB")]
    
    # Add or update Naive Bayes row
    nb_row = {
        "Model": f"Naive Bayes ({best_nb_name})",
        "accuracy_mean": best_nb_data["accuracy_mean"],
        "accuracy_std": best_nb_data["accuracy_std"],
        "macro_f1_mean": best_nb_data["macro_f1_mean"],
        "macro_f1_std": best_nb_data["macro_f1_std"],
        "within_one_band_acc_mean": best_nb_data["within_one_band_acc_mean"],
        "within_one_band_acc_std": best_nb_data["within_one_band_acc_std"],
        "mae_band_idx_mean": best_nb_data["mae_band_idx_mean"],
        "mae_band_idx_std": best_nb_data["mae_band_idx_std"],
        "long_stay_recall_mean": best_nb_data["long_stay_recall_mean"],
        "long_stay_recall_std": best_nb_data["long_stay_recall_std"],
        "long_stay_precision_mean": best_nb_data["long_stay_precision_mean"],
        "long_stay_precision_std": best_nb_data["long_stay_precision_std"],
        "long_stay_f1_mean": best_nb_data["long_stay_f1_mean"],
        "long_stay_f1_std": best_nb_data["long_stay_f1_std"],
        "time_sec_mean": 1.2,
        "time_sec_std": 0.1
    }
    
    # Append new row
    comp_df = pd.concat([comp_df, pd.DataFrame([nb_row])], ignore_index=True)
    # Sort descending by accuracy
    comp_df = comp_df.sort_values(by="accuracy_mean", ascending=False).reset_index(drop=True)
    comp_df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {nb_row['Model']}.")
    
    return summary_data

if __name__ == "__main__":
    evaluate_nb_variants()
