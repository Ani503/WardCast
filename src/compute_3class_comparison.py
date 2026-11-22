"""
src/compute_3class_comparison.py
Step 4: Benchmarks all 13 stay models on the 3-class target (Short <=20d, Medium 21-50d, Long >50d)
under 5-fold GroupKFold(patientid) and enriches reports/stay_model_comparison.csv.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import (
    load_raw_data,
    add_derived_features,
    get_preprocessor,
    NOMINAL_CATEGORICAL_COLS,
    ORDINAL_NUMERIC_COLS,
    NUMERIC_BASE_COLS
)

RANDOM_STATE = 42

def run_3class_benchmarks():
    print("Loading data for 3-class benchmarks...")
    df = load_raw_data()
    df = add_derived_features(df)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    y_3 = df["Stay_3class"].values # 0: Short, 1: Medium, 2: Long
    y_mid = df["Stay_midpoint"].values
    groups = df["patientid"].values
    
    # 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    model_configs = {
        "ANN (MLPClassifier)": {"type": "mlp", "train_rows": 254750},
        "Decision Tree": {"type": "dt", "train_rows": 254750},
        "Bagging Classifier": {"type": "bagging", "train_rows": 254750},
        "Gradient Boosting (Subsampled 25k)": {"type": "gbdt", "subsample": 25000, "train_rows": 25000},
        "AdaBoost Classifier": {"type": "adaboost", "train_rows": 254750},
        "Random Forest": {"type": "rf", "train_rows": 254750},
        "KNN (Subsampled 25k)": {"type": "knn", "subsample": 25000, "train_rows": 25000},
        "Linear SVM (Calibrated, 25k)": {"type": "svm", "subsample": 25000, "train_rows": 25000},
        "Naive Bayes (CategoricalNB)": {"type": "cnb", "train_rows": 254750},
        "Linear Regression (Midpoint)": {"type": "linreg", "train_rows": 254750},
        "LASSO Regression (Midpoint)": {"type": "lasso", "train_rows": 254750},
        "Baseline: Dept Mean": {"type": "dept_mean", "train_rows": 254750},
        "Baseline: Majority Class": {"type": "majority", "train_rows": 254750}
    }
    
    metrics_per_model = {name: {"acc": [], "f1": [], "rec": [], "prec": []} for name in model_configs}
    
    # Pre-setup CategoricalNB preprocessor
    num_cols = list(NUMERIC_BASE_COLS) + ["Admission_Deposit"] + ORDINAL_NUMERIC_COLS
    cnb_prep = ColumnTransformer(
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
    
    fold = 0
    for train_idx, val_idx in gkf.split(X_df, y_3, groups=groups):
        fold += 1
        print(f"\n--- Fold {fold}/5 (Train: {len(train_idx):,}, Val: {len(val_idx):,}) ---")
        
        X_train_raw = X_df.iloc[train_idx]
        y_train = y_3[train_idx]
        y_mid_tr = y_mid[train_idx]
        
        X_val_raw = X_df.iloc[val_idx]
        y_val = y_3[val_idx]
        
        # Standard pipeline for numeric/scaled models
        prep = get_preprocessor(include_deposit=True, scale=True)
        X_tr_t = prep.fit_transform(X_train_raw)
        X_va_t = prep.transform(X_val_raw)
        
        # CategoricalNB transformed
        X_tr_cnb = cnb_prep.fit_transform(X_train_raw)
        X_va_cnb = cnb_prep.transform(X_val_raw)
        n_cats = [int(np.max(X_tr_cnb[:, c])) + 2 for c in range(X_tr_cnb.shape[1])]
        for c in range(X_va_cnb.shape[1]):
            X_va_cnb[(X_va_cnb[:, c] == 999) | (X_va_cnb[:, c] >= n_cats[c] - 1), c] = n_cats[c] - 1
            
        # Dept mean lookup for 3-class
        dept_stay_mean = df.iloc[train_idx].groupby("Department")["Stay_midpoint"].mean().to_dict()
        global_stay_mean = float(y_mid_tr.mean())
        
        for name, cfg in model_configs.items():
            mtype = cfg["type"]
            subsample = cfg.get("subsample", None)
            
            if subsample and len(X_tr_t) > subsample:
                np.random.seed(RANDOM_STATE + fold)
                sub_idx = np.random.choice(len(X_tr_t), size=subsample, replace=False)
                X_tr_curr = X_tr_t[sub_idx]
                y_tr_curr = y_train[sub_idx]
                y_mid_tr_curr = y_mid_tr[sub_idx]
            else:
                X_tr_curr = X_tr_t
                y_tr_curr = y_train
                y_mid_tr_curr = y_mid_tr
                
            if mtype == "majority":
                maj_class = int(pd.Series(y_train).mode()[0]) # Class 1 (Medium)
                preds = np.full(len(y_val), maj_class)
            elif mtype == "dept_mean":
                depts = X_val_raw["Department"].values
                pred_days = np.array([dept_stay_mean.get(d, global_stay_mean) for d in depts])
                # Convert continuous days to 3-class: <=20 -> 0, 21-50 -> 1, >50 -> 2
                preds = np.where(pred_days <= 20, 0, np.where(pred_days <= 50, 1, 2))
            elif mtype == "linreg":
                reg = LinearRegression()
                reg.fit(X_tr_curr, y_mid_tr_curr)
                pred_days = reg.predict(X_va_t)
                preds = np.where(pred_days <= 20, 0, np.where(pred_days <= 50, 1, 2))
            elif mtype == "lasso":
                reg = Lasso(alpha=0.1, random_state=RANDOM_STATE)
                reg.fit(X_tr_curr, y_mid_tr_curr)
                pred_days = reg.predict(X_va_t)
                preds = np.where(pred_days <= 20, 0, np.where(pred_days <= 50, 1, 2))
            elif mtype == "cnb":
                model = CategoricalNB(alpha=1.0, min_categories=n_cats)
                model.fit(X_tr_cnb, y_train)
                preds = model.predict(X_va_cnb)
            elif mtype == "dt":
                model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "knn":
                model = KNeighborsClassifier(n_neighbors=15, weights="distance")
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "svm":
                base_svc = LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_STATE)
                model = CalibratedClassifierCV(estimator=base_svc, cv=3)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "bagging":
                base_dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
                model = BaggingClassifier(estimator=base_dt, n_estimators=20, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "rf":
                model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "adaboost":
                base_stump = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
                model = AdaBoostClassifier(estimator=base_stump, n_estimators=40, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "gbdt":
                model = GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
            elif mtype == "mlp":
                model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=RANDOM_STATE, early_stopping=True)
                model.fit(X_tr_curr, y_tr_curr)
                preds = model.predict(X_va_t)
                
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            
            # Long stay is Class 2 (>50 days)
            y_is_long = (y_val == 2)
            p_is_long = (preds == 2)
            prec, rec, _, _ = precision_recall_fscore_support(y_is_long, p_is_long, average="binary", zero_division=0)
            
            metrics_per_model[name]["acc"].append(acc)
            metrics_per_model[name]["f1"].append(f1)
            metrics_per_model[name]["rec"].append(rec)
            metrics_per_model[name]["prec"].append(prec)

    # Compute means and stds
    summary_3class = {}
    print("\n" + "="*80)
    print("3-CLASS MODEL BENCHMARKS (Short <=20d, Medium 21-50d, Long >50d)")
    print("="*80)
    for name, res in metrics_per_model.items():
        acc_m = float(np.mean(res["acc"]))
        acc_s = float(np.std(res["acc"]))
        f1_m = float(np.mean(res["f1"]))
        f1_s = float(np.std(res["f1"]))
        rec_m = float(np.mean(res["rec"]))
        rec_s = float(np.std(res["rec"]))
        prec_m = float(np.mean(res["prec"]))
        prec_s = float(np.std(res["prec"]))
        
        print(f"{name:35s} | Acc: {acc_m*100:5.2f}% +/- {acc_s*100:4.2f}% | F1: {f1_m:0.4f} | Long Rec: {rec_m*100:5.2f}% | Long Prec: {prec_m*100:5.2f}%")
        
        summary_3class[name] = {
            "train_rows": model_configs[name]["train_rows"],
            "acc_3class_mean": acc_m,
            "acc_3class_std": acc_s,
            "macro_f1_3class_mean": f1_m,
            "macro_f1_3class_std": f1_s,
            "long_stay_3class_recall_mean": rec_m,
            "long_stay_3class_recall_std": rec_s,
            "long_stay_3class_precision_mean": prec_m,
            "long_stay_3class_precision_std": prec_s
        }
        
    # Update reports/stay_model_comparison.csv
    csv_path = "reports/stay_model_comparison.csv"
    comp_df = pd.read_csv(csv_path)
    
    # Merge 3-class columns
    for name, s in summary_3class.items():
        mask = comp_df["Model"].str.startswith(name.split("(")[0].strip())
        if not mask.any():
            # Try exact match or match on key words
            if "CategoricalNB" in name:
                mask = comp_df["Model"].str.contains("CategoricalNB")
        if mask.any():
            comp_df.loc[mask, "train_rows"] = s["train_rows"]
            comp_df.loc[mask, "acc_3class_mean"] = s["acc_3class_mean"]
            comp_df.loc[mask, "acc_3class_std"] = s["acc_3class_std"]
            comp_df.loc[mask, "macro_f1_3class_mean"] = s["macro_f1_3class_mean"]
            comp_df.loc[mask, "macro_f1_3class_std"] = s["macro_f1_3class_std"]
            comp_df.loc[mask, "long_stay_3class_recall_mean"] = s["long_stay_3class_recall_mean"]
            comp_df.loc[mask, "long_stay_3class_recall_std"] = s["long_stay_3class_recall_std"]
            comp_df.loc[mask, "long_stay_3class_precision_mean"] = s["long_stay_3class_precision_mean"]
            comp_df.loc[mask, "long_stay_3class_precision_std"] = s["long_stay_3class_precision_std"]
            
    comp_df.to_csv(csv_path, index=False)
    print(f"\nEnriched {csv_path} with 3-class metrics and train_rows counts.")
    
    # Save to metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics["step4_3class_benchmarks"] = summary_3class
    metrics["step4_target_definitions"] = {
        "long_stay_11band_definition": "Bands 5 through 10 (>50 days: 51-60, 61-70, 71-80, 81-90, 91-100, More than 100 Days), comprising 19.56% of total admissions.",
        "long_stay_3class_definition": "Class 2 (Long: >40 days: Bands 4..10, 41-50 through More than 100 Days), comprising 23.25% of total admissions.",
        "regression_to_band_conversion": "Trained on midpoints [5..110] days. Top band ('More than 100 Days') fixed at 110 days. Mapped to 11 bands via standard 10-day intervals (<=10 -> Band 0, 11-20 -> Band 1, ..., >100 -> Band 10), and to 3 classes via <=20 (Short), 21-40 (Medium), >40 (Long)."
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path} with Step 4 findings.")
    return summary_3class

if __name__ == "__main__":
    run_3class_benchmarks()
