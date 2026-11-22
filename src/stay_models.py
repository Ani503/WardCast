"""
src/stay_models.py
Phase 2: Rigorous evaluation of all 9 syllabus classification models,
baselines, regression track, feature selection, imbalance handling,
and ablation studies on Kaggle hospital length of stay data.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

# Syllabus Models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

from src.prep import (
    load_raw_data,
    add_derived_features,
    get_preprocessor,
    get_iqr_outlier_mask,
    STAY_BANDS,
    BAND_TO_LABEL,
    BAND_TO_MIDPOINT,
    CLASS_3_NAMES
)

REPORTS_DIR = "reports"
MODELS_DIR = "models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the most frequent class seen during fit."""
    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.majority_class_ = vals[np.argmax(counts)]
        self.classes_ = np.sort(vals)
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.majority_class_, dtype=int)

    def predict_proba(self, X):
        probas = np.zeros((len(X), len(self.classes_)))
        probas[:, self.majority_class_] = 1.0
        return probas

class DepartmentMeanStayBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the rounded mean stay band index per hospital department."""
    def __init__(self):
        self.dept_modes_ = {}
        self.global_mode_ = 2

    def fit(self, X_df, y):
        self.global_mode_ = int(pd.Series(y).mode()[0])
        depts = X_df["Department"].values
        for dept in np.unique(depts):
            mask = depts == dept
            if mask.sum() > 0:
                self.dept_modes_[dept] = int(pd.Series(y[mask]).mode()[0])
            else:
                self.dept_modes_[dept] = self.global_mode_
        return self

    def predict(self, X_df):
        depts = X_df["Department"].values
        return np.array([self.dept_modes_.get(d, self.global_mode_) for d in depts], dtype=int)

    def predict_proba(self, X_df):
        preds = self.predict(X_df)
        probas = np.zeros((len(preds), len(STAY_BANDS)))
        for i, p in enumerate(preds):
            probas[i, p] = 1.0
        return probas

class RegressionBandClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a continuous regressor to output discrete band predictions."""
    def __init__(self, regressor):
        self.regressor = regressor
        self.midpoints_ = np.array([BAND_TO_MIDPOINT[b] for b in STAY_BANDS])

    def fit(self, X, y_midpoint):
        self.regressor.fit(X, y_midpoint)
        self.classes_ = np.arange(len(STAY_BANDS))
        return self

    def predict(self, X):
        continuous_pred = self.regressor.predict(X)
        # Find nearest midpoint
        diffs = np.abs(continuous_pred[:, np.newaxis] - self.midpoints_)
        return np.argmin(diffs, axis=1)

    def predict_proba(self, X):
        continuous_pred = self.regressor.predict(X)
        diffs = np.abs(continuous_pred[:, np.newaxis] - self.midpoints_)
        # Softmax over negative distances
        scores = -diffs / 10.0
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Computes comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    mae = mean_absolute_error(y_true, y_pred)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # Long-stay class is class index 10 ('More than 100 Days')
    long_stay_rec = report.get("10", {}).get("recall", 0.0)
    long_stay_prec = report.get("10", {}).get("precision", 0.0)
    long_stay_f1 = report.get("10", {}).get("f1-score", 0.0)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "within_one_band_acc": within_one,
        "mae_band_idx": mae,
        "long_stay_recall": long_stay_rec,
        "long_stay_precision": long_stay_prec,
        "long_stay_f1": long_stay_f1
    }

def run_cross_validation():
    print("="*60)
    print("Starting Phase 2: Stay Models Cross-Validation (GroupKFold=5)")
    print("="*60)
    
    df_raw = load_raw_data()
    df = add_derived_features(df_raw)
    
    groups = df["patientid"].values
    y = df["Stay_label"].values
    y_mid = df["Stay_midpoint"].values
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    
    gkf = GroupKFold(n_splits=5)
    
    # Define models dictionary
    # Subsampled models use stratified 25k subset of training fold for tractability
    model_configs = {
        "Baseline: Majority Class": {"type": "majority"},
        "Baseline: Dept Mean": {"type": "dept_mean"},
        "Linear Regression (Midpoint)": {"type": "linreg"},
        "LASSO Regression (Midpoint)": {"type": "lasso"},
        "Naive Bayes (GaussianNB)": {"type": "nb"},
        "Decision Tree": {"type": "dt"},
        "KNN (Subsampled 25k)": {"type": "knn", "subsample": 25000},
        "Linear SVM (Calibrated, 25k)": {"type": "svm", "subsample": 25000},
        "Bagging Classifier": {"type": "bagging"},
        "Random Forest": {"type": "rf"},
        "AdaBoost Classifier": {"type": "adaboost"},
        "Gradient Boosting (Subsampled 25k)": {"type": "gbdt", "subsample": 25000},
        "ANN (MLPClassifier)": {"type": "mlp"},
    }
    
    fold_results = {name: [] for name in model_configs}
    oof_predictions = {name: np.zeros(len(df), dtype=int) for name in model_configs}
    oof_probabilities = {name: np.zeros((len(df), len(STAY_BANDS))) for name in model_configs}
    
    fold_idx = 0
    for train_idx, val_idx in gkf.split(X_df, y, groups=groups):
        fold_idx += 1
        print(f"\n--- Fold {fold_idx}/5 (Train: {len(train_idx):,}, Val: {len(val_idx):,}) ---")
        
        # Verify zero patient overlap
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"FATAL: Patient leakage detected in fold {fold_idx}!"
        
        X_train_fold = X_df.iloc[train_idx]
        y_train_fold = y[train_idx]
        y_mid_train_fold = y_mid[train_idx]
        
        X_val_fold = X_df.iloc[val_idx]
        y_val_fold = y[val_idx]
        y_mid_val_fold = y_mid[val_idx]
        
        # Preprocessor fit purely on train fold
        preprocessor = get_preprocessor(include_deposit=True, scale=True)
        X_tr_trans = preprocessor.fit_transform(X_train_fold)
        X_va_trans = preprocessor.transform(X_val_fold)
        
        for name, cfg in model_configs.items():
            t0 = time.time()
            mtype = cfg["type"]
            subsample = cfg.get("subsample", None)
            
            # Subsample if needed for heavy models
            if subsample and len(X_tr_trans) > subsample:
                np.random.seed(RANDOM_STATE + fold_idx)
                # Stratified sample by y_train_fold
                sub_indices = []
                for cls in np.unique(y_train_fold):
                    cls_idx = np.where(y_train_fold == cls)[0]
                    cls_sample_n = max(5, int(len(cls_idx) / len(y_train_fold) * subsample))
                    sampled = np.random.choice(cls_idx, size=min(cls_sample_n, len(cls_idx)), replace=False)
                    sub_indices.extend(sampled)
                sub_indices = np.array(sub_indices)
                X_tr_curr = X_tr_trans[sub_indices]
                y_tr_curr = y_train_fold[sub_indices]
                y_mid_tr_curr = y_mid_train_fold[sub_indices]
            else:
                X_tr_curr = X_tr_trans
                y_tr_curr = y_train_fold
                y_mid_tr_curr = y_mid_train_fold
                
            # Instantiate model
            if mtype == "majority":
                model = MajorityClassBaseline()
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "dept_mean":
                model = DepartmentMeanStayBaseline()
                model.fit(X_train_fold, y_tr_curr)
                val_pred = model.predict(X_val_fold)
                val_prob = model.predict_proba(X_val_fold)
            elif mtype == "linreg":
                reg = LinearRegression()
                model = RegressionBandClassifier(reg)
                model.fit(X_tr_curr, y_mid_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "lasso":
                reg = Lasso(alpha=0.1, random_state=RANDOM_STATE)
                model = RegressionBandClassifier(reg)
                model.fit(X_tr_curr, y_mid_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "nb":
                model = GaussianNB()
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "dt":
                # Inner-fold tuned depth=12
                model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "knn":
                model = KNeighborsClassifier(n_neighbors=15, weights="distance")
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "svm":
                base_svc = LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_STATE)
                model = CalibratedClassifierCV(estimator=base_svc, cv=3)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "bagging":
                base_dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
                model = BaggingClassifier(estimator=base_dt, n_estimators=20, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "rf":
                model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "adaboost":
                base_stump = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
                model = AdaBoostClassifier(estimator=base_stump, n_estimators=40, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "gbdt":
                model = GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            elif mtype == "mlp":
                model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, alpha=0.01, early_stopping=True, random_state=RANDOM_STATE)
                model.fit(X_tr_curr, y_tr_curr)
                val_pred = model.predict(X_va_trans)
                val_prob = model.predict_proba(X_va_trans)
            else:
                raise ValueError(f"Unknown model type: {mtype}")
                
            elapsed = time.time() - t0
            metrics = evaluate_predictions(y_val_fold, val_pred)
            metrics["time_sec"] = elapsed
            fold_results[name].append(metrics)
            
            oof_predictions[name][val_idx] = val_pred
            oof_probabilities[name][val_idx] = val_prob
            
            print(f"  {name:<35} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f} | Within-1: {metrics['within_one_band_acc']:.4f} | Time: {elapsed:.1f}s")

    # Aggregate cross-validation results
    summary_rows = []
    for name, res_list in fold_results.items():
        row = {"Model": name}
        for metric_key in ["accuracy", "macro_f1", "within_one_band_acc", "mae_band_idx", "long_stay_recall", "long_stay_precision", "long_stay_f1", "time_sec"]:
            vals = [r[metric_key] for r in res_list]
            row[f"{metric_key}_mean"] = np.mean(vals)
            row[f"{metric_key}_std"] = np.std(vals)
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by="macro_f1_mean", ascending=False, inplace=True)
    summary_path = os.path.join(REPORTS_DIR, "stay_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved cross-validation summary to {summary_path}")
    
    # Save confusion matrix plot for best model
    best_model_name = summary_df.iloc[0]["Model"]
    print(f"\nTop performing model: {best_model_name}")
    best_oof_pred = oof_predictions[best_model_name]
    cm = confusion_matrix(y, best_oof_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {best_model_name} (5-Fold Out-of-Fold)")
    plt.colorbar()
    tick_marks = np.arange(len(STAY_BANDS))
    plt.xticks(tick_marks, STAY_BANDS, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, STAY_BANDS, fontsize=9)
    plt.ylabel("True Stay Band")
    plt.xlabel("Predicted Stay Band")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Run ablation: Imbalance handling comparison on Random Forest
    print("\n--- Running Imbalance Comparison (Random Forest) ---")
    run_imbalance_comparison(X_df, y, groups)
    
    # Run ablation: Feature Selection with LASSO and Mutual Information / F-classif
    print("\n--- Running Feature Selection Analysis ---")
    feat_sel_res = run_feature_selection(X_df, y, groups)
    
    # Run ablation: With vs Without Admission_Deposit & IQR Outlier Removal
    print("\n--- Running Ablation Studies (Deposit & Outliers) ---")
    ablation_res = run_ablations(X_df, y, groups)
    
    # Fit final pipeline on all data and save to models/stay_model.joblib
    # Fit final pipeline on all data and save to models/stay_model.joblib
    print("\n--- Training and Saving Final Pipeline (ANN MLPClassifier) ---")
    final_preprocessor = get_preprocessor(include_deposit=True, scale=True)
    final_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=40,
        alpha=0.01,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    final_pipeline = Pipeline(steps=[
        ("preprocessor", final_preprocessor),
        ("classifier", final_mlp)
    ])
    final_pipeline.fit(X_df, y)
    
    model_save_path = os.path.join(MODELS_DIR, "stay_model.joblib")
    joblib.dump(final_pipeline, model_save_path)
    print(f"Saved final pipeline to {model_save_path}")

def predict_patient_band_probabilities(patient_data, pipeline=None):
    """
    Returns a dictionary of length-of-stay band probabilities for a patient.
    Accepts a dictionary of patient features.
    """
    if pipeline is None:
        pipeline = joblib.load(os.path.join(MODELS_DIR, "stay_model.joblib"))
    if isinstance(patient_data, dict):
        df_p = pd.DataFrame([patient_data])
    else:
        df_p = pd.DataFrame(patient_data)
    df_p = add_derived_features(df_p)
    
    feature_cols = [c for c in df_p.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    probs = pipeline.predict_proba(df_p[feature_cols])[0]
    return {band: float(probs[i]) for i, band in enumerate(STAY_BANDS)}

    
    # Update metrics.json
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["stay_prediction"] = {
        "best_model": best_model_name,
        "best_accuracy_mean": float(summary_df.iloc[0]["accuracy_mean"]),
        "best_accuracy_std": float(summary_df.iloc[0]["accuracy_std"]),
        "best_macro_f1_mean": float(summary_df.iloc[0]["macro_f1_mean"]),
        "best_macro_f1_std": float(summary_df.iloc[0]["macro_f1_std"]),
        "best_within_one_band_mean": float(summary_df.iloc[0]["within_one_band_acc_mean"]),
        "best_mae_band_idx_mean": float(summary_df.iloc[0]["mae_band_idx_mean"]),
        "best_long_stay_recall_mean": float(summary_df.iloc[0]["long_stay_recall_mean"]),
        "best_long_stay_precision_mean": float(summary_df.iloc[0]["long_stay_precision_mean"]),
        "baseline_majority_accuracy": float(summary_df[summary_df["Model"] == "Baseline: Majority Class"]["accuracy_mean"].values[0]),
        "baseline_majority_macro_f1": float(summary_df[summary_df["Model"] == "Baseline: Majority Class"]["macro_f1_mean"].values[0]),
        "baseline_dept_accuracy": float(summary_df[summary_df["Model"] == "Baseline: Dept Mean"]["accuracy_mean"].values[0]),
        "baseline_dept_macro_f1": float(summary_df[summary_df["Model"] == "Baseline: Dept Mean"]["macro_f1_mean"].values[0]),
        "feature_selection": feat_sel_res,
        "ablations": ablation_res
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path} with Phase 2 results.")

def run_imbalance_comparison(X_df, y, groups):
    gkf = GroupKFold(n_splits=3)
    unweighted_f1, weighted_f1, resampled_f1 = [], [], []
    
    for train_idx, val_idx in gkf.split(X_df, y, groups=groups):
        X_tr = X_df.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X_df.iloc[val_idx]
        y_val = y[val_idx]
        
        prep = get_preprocessor(include_deposit=True, scale=True)
        X_tr_trans = prep.fit_transform(X_tr)
        X_va_trans = prep.transform(X_va)
        
        # 1. Unweighted Random Forest (depth 12 for fast check)
        rf_unw = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        rf_unw.fit(X_tr_trans, y_tr)
        unweighted_f1.append(f1_score(y_val, rf_unw.predict(X_va_trans), average="macro", zero_division=0))
        
        # 2. Balanced Class Weight
        rf_w = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        rf_w.fit(X_tr_trans, y_tr)
        weighted_f1.append(f1_score(y_val, rf_w.predict(X_va_trans), average="macro", zero_division=0))
        
        # 3. Resampling inside training fold
        # Oversample minority classes (<5% prevalence)
        X_res_list = [X_tr_trans]
        y_res_list = [y_tr]
        classes, counts = np.unique(y_tr, return_counts=True)
        target_count = int(np.median(counts))
        for c, count in zip(classes, counts):
            if count < target_count * 0.5:
                idx = np.where(y_tr == c)[0]
                X_c_res, y_c_res = resample(X_tr_trans[idx], y_tr[idx], replace=True, n_samples=target_count - count, random_state=RANDOM_STATE)
                X_res_list.append(X_c_res)
                y_res_list.append(y_c_res)
        X_tr_res = np.vstack(X_res_list)
        y_tr_res = np.concatenate(y_res_list)
        
        rf_res = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        rf_res.fit(X_tr_res, y_tr_res)
        resampled_f1.append(f1_score(y_val, rf_res.predict(X_va_trans), average="macro", zero_division=0))
        
    print(f"Imbalance Comparison (Macro-F1):")
    print(f"  Unweighted:  {np.mean(unweighted_f1):.4f} +/- {np.std(unweighted_f1):.4f}")
    print(f"  Class-Weight: {np.mean(weighted_f1):.4f} +/- {np.std(weighted_f1):.4f}")
    print(f"  Resampled:   {np.mean(resampled_f1):.4f} +/- {np.std(resampled_f1):.4f}")
    return {
        "unweighted_macro_f1": float(np.mean(unweighted_f1)),
        "class_weighted_macro_f1": float(np.mean(weighted_f1)),
        "resampled_macro_f1": float(np.mean(resampled_f1))
    }

def run_feature_selection(X_df, y, groups):
    # Using 3 folds for feature selection curve
    gkf = GroupKFold(n_splits=3)
    k_values = [20, 40, 60, 80, 100, 116]
    k_scores = {k: [] for k in k_values}
    
    for train_idx, val_idx in gkf.split(X_df, y, groups=groups):
        X_tr = X_df.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X_df.iloc[val_idx]
        y_val = y[val_idx]
        
        prep = get_preprocessor(include_deposit=True, scale=True)
        X_tr_trans = prep.fit_transform(X_tr)
        X_va_trans = prep.transform(X_va)
        
        for k in k_values:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_tr_trans.shape[1]))
            X_tr_sel = selector.fit_transform(X_tr_trans, y_tr)
            X_va_sel = selector.transform(X_va_trans)
            
            clf = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
            clf.fit(X_tr_sel, y_tr)
            score = f1_score(y_val, clf.predict(X_va_sel), average="macro", zero_division=0)
            k_scores[k].append(score)
            
    avg_scores = {k: float(np.mean(k_scores[k])) for k in k_values}
    full_score = avg_scores[116]
    smallest_k = min([k for k, s in avg_scores.items() if s >= full_score - 0.01])
    
    print(f"Feature Selection curve (k -> macro-F1): {avg_scores}")
    print(f"Smallest feature set within 1% of full-set macro-F1: k={smallest_k} (F1: {avg_scores[smallest_k]:.4f} vs full {full_score:.4f})")
    return {
        "k_scores": avg_scores,
        "smallest_k_within_1_percent": smallest_k,
        "smallest_k_f1": avg_scores[smallest_k],
        "full_f1": full_score
    }

def run_ablations(X_df, y, groups):
    gkf = GroupKFold(n_splits=3)
    scores_full = []
    scores_no_deposit = []
    scores_outlier_removed = []
    
    for train_idx, val_idx in gkf.split(X_df, y, groups=groups):
        X_tr = X_df.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X_df.iloc[val_idx]
        y_val = y[val_idx]
        
        # 1. Full with deposit
        prep_full = get_preprocessor(include_deposit=True, scale=True)
        X_tr_full = prep_full.fit_transform(X_tr)
        X_va_full = prep_full.transform(X_va)
        rf1 = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        rf1.fit(X_tr_full, y_tr)
        scores_full.append(f1_score(y_val, rf1.predict(X_va_full), average="macro", zero_division=0))
        
        # 2. Without Admission_Deposit
        X_tr_nodep = X_tr.drop(columns=["Admission_Deposit"])
        X_va_nodep = X_va.drop(columns=["Admission_Deposit"])
        prep_nodep = get_preprocessor(include_deposit=False, scale=True)
        X_tr_nodep_tr = prep_nodep.fit_transform(X_tr_nodep)
        X_va_nodep_tr = prep_nodep.transform(X_va_nodep)
        rf2 = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        rf2.fit(X_tr_nodep_tr, y_tr)
        scores_no_deposit.append(f1_score(y_val, rf2.predict(X_va_nodep_tr), average="macro", zero_division=0))
        
        # 3. With 1.5x IQR Outlier removal on training fold
        outlier_mask = get_iqr_outlier_mask(X_tr)
        X_tr_clean = X_tr[~outlier_mask]
        y_tr_clean = y_tr[~outlier_mask]
        prep_clean = get_preprocessor(include_deposit=True, scale=True)
        X_tr_clean_tr = prep_clean.fit_transform(X_tr_clean)
        X_va_clean_tr = prep_clean.transform(X_va)
        rf3 = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        rf3.fit(X_tr_clean_tr, y_tr_clean)
        scores_outlier_removed.append(f1_score(y_val, rf3.predict(X_va_clean_tr), average="macro", zero_division=0))
        
    res = {
        "full_with_deposit_macro_f1": float(np.mean(scores_full)),
        "without_deposit_macro_f1": float(np.mean(scores_no_deposit)),
        "outlier_removed_macro_f1": float(np.mean(scores_outlier_removed))
    }
    print(f"Ablation Results:")
    print(f"  Full (with deposit):     {res['full_with_deposit_macro_f1']:.4f}")
    print(f"  Without deposit:         {res['without_deposit_macro_f1']:.4f} (diff: {res['without_deposit_macro_f1'] - res['full_with_deposit_macro_f1']:+.4f})")
    print(f"  Training Outliers Clean: {res['outlier_removed_macro_f1']:.4f} (diff: {res['outlier_removed_macro_f1'] - res['full_with_deposit_macro_f1']:+.4f})")
    return res

if __name__ == "__main__":
    run_cross_validation()
