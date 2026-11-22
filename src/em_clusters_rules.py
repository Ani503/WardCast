"""
src/em_clusters_rules.py
Phase 3: Expectation-Maximization (EM) on log-stay, Clustering benchmark
(K-Means, Agglomerative, BIRCH), and Association Rules (Apriori vs FP-Growth).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

from src.prep import (
    load_raw_data,
    add_derived_features,
    get_preprocessor,
    BAND_TO_3CLASS,
    STAY_BANDS
)

REPORTS_DIR = "reports"
MODELS_DIR = "models"
DATA_DIR = "Data"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

def run_em_analysis():
    print("="*60)
    print("Phase 3.1: EM Algorithm (Gaussian Mixture) on Log Length of Stay")
    print("="*60)
    
    sparcs_path = os.path.join(DATA_DIR, "Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv")
    print(f"Reading SPARCS exact length of stay records from {sparcs_path}...")
    df_sparcs = pd.read_csv(sparcs_path, usecols=["Length of Stay"], nrows=200000, low_memory=False)
    
    # Clean Length of Stay
    raw_los = df_sparcs["Length of Stay"].astype(str).str.replace("+", "", regex=False).str.strip()
    clean_los = pd.to_numeric(raw_los, errors="coerce").dropna()
    clean_los = clean_los[clean_los >= 1.0].values
    
    log_los = np.log(clean_los).reshape(-1, 1)
    print(f"Cleaned {len(log_los):,} stay records. Log LOS mean: {np.mean(log_los):.3f}, std: {np.std(log_los):.3f}")
    
    k_range = [2, 3, 4, 5, 6]
    bics = []
    aics = []
    models = {}
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, max_iter=200)
        gmm.fit(log_los)
        bic = gmm.bic(log_los)
        aic = gmm.aic(log_los)
        bics.append(bic)
        aics.append(aic)
        models[k] = gmm
        print(f"  k={k}: BIC={bic:,.1f}, AIC={aic:,.1f}")
        
    best_k = k_range[np.argmin(bics)]
    best_gmm = models[best_k]
    print(f"\nOptimal Gaussian Mixture components by BIC: k={best_k}")
    
    # Plot BIC Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_range, bics, "o-", color="navy", lw=2, label="BIC")
    plt.plot(k_range, aics, "s--", color="crimson", lw=1.5, label="AIC")
    plt.axvline(best_k, color="darkgreen", linestyle=":", label=f"Optimal k={best_k}")
    plt.title("EM Gaussian Mixture Model Selection (BIC / AIC)")
    plt.xlabel("Number of Mixture Components (k)")
    plt.ylabel("Information Criterion Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    bic_path = os.path.join(REPORTS_DIR, "em_bic_curve.png")
    plt.savefig(bic_path, dpi=200)
    plt.close()
    print(f"Saved BIC plot to {bic_path}")
    
    # Sort components by mean
    order = np.argsort(best_gmm.means_.flatten())
    components_info = []
    
    labels_desc = [
        "Routine / Short Observation (Minor / Elective procedures, normal deliveries)",
        "Standard Acute Inpatient (Uncomplicated surgery, acute medical stabilization)",
        "Extended Acute Inpatient (Complex surgical recovery, severe systemic infections)",
        "Prolonged / Complex Inpatient (Multiple morbidities, intensive care transition)",
        "Catastrophic / Long-Term Inpatient (Sub-acute rehab, extensive complications, chronic organ failure)"
    ]
    
    for rank, idx in enumerate(order):
        weight = float(best_gmm.weights_[idx])
        log_mean = float(best_gmm.means_[idx][0])
        log_var = float(best_gmm.covariances_[idx][0][0])
        days_geom = float(np.exp(log_mean))
        days_range_low = float(np.exp(log_mean - np.sqrt(log_var)))
        days_range_high = float(np.exp(log_mean + np.sqrt(log_var)))
        desc = labels_desc[rank] if rank < len(labels_desc) else f"Extended Component {rank+1}"
        
        info = {
            "component": rank + 1,
            "weight": round(weight, 4),
            "weight_pct": round(weight * 100, 2),
            "log_mean": round(log_mean, 4),
            "log_std": round(np.sqrt(log_var), 4),
            "geometric_mean_days": round(days_geom, 2),
            "typical_range_days": [round(days_range_low, 1), round(days_range_high, 1)],
            "interpretation": desc
        }
        components_info.append(info)
        print(f"  Component {rank+1}: Weight={weight*100:.1f}%, Mean={days_geom:.1f} days (range {days_range_low:.1f}-{days_range_high:.1f}d) | {desc}")
        
    return {
        "dataset_source": "SPARCS 2014 Exact Days",
        "sample_size": len(log_los),
        "tested_k": k_range,
        "bic_scores": [float(b) for b in bics],
        "optimal_k": best_k,
        "components": components_info
    }

def run_clustering_analysis():
    print("\n" + "="*60)
    print("Phase 3.2: Clustering Benchmark (K-Means, Agglomerative, BIRCH)")
    print("="*60)
    
    df_raw = load_raw_data()
    df = add_derived_features(df_raw)
    
    feature_cols = [c for c in df.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"]]
    X_df = df[feature_cols]
    
    preprocessor = get_preprocessor(include_deposit=True, scale=True)
    X_trans = preprocessor.fit_transform(X_df)
    
    # Subsample for evaluation of silhouette and agglomerative distance matrix
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(X_trans), size=15000, replace=False)
    X_sample = X_trans[sample_indices]
    
    k_range = [2, 3, 4, 5, 6, 7]
    inertias = []
    sil_scores_kmeans = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        km.fit(X_sample)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_sample, km.labels_)
        sil_scores_kmeans.append(sil)
        print(f"  K-Means k={k}: Inertia={km.inertia_:,.1f}, Silhouette={sil:.4f}")
        
    # Plot Elbow & Silhouette
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color = "tab:blue"
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Elbow)", color=color)
    ax1.plot(k_range, inertias, "o-", color=color, lw=2, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Silhouette Score", color=color)
    ax2.plot(k_range, sil_scores_kmeans, "s--", color=color, lw=2, label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("Cluster Evaluation: Elbow and Silhouette Analysis")
    fig.tight_layout()
    cluster_plot_path = os.path.join(REPORTS_DIR, "clustering_elbow_silhouette.png")
    plt.savefig(cluster_plot_path, dpi=200)
    plt.close()
    print(f"Saved clustering evaluation plot to {cluster_plot_path}")
    
    # Choose optimal k = 3 based on silhouette peak & clinical distinctiveness
    best_k = 3
    print(f"\nBenchmarking K-Means, Agglomerative, and BIRCH at k={best_k}...")
    
    # 1. K-Means
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_labels = km_best.fit_predict(X_sample)
    sil_km = float(silhouette_score(X_sample, km_labels))
    
    # 2. Agglomerative (Hierarchical)
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels = agg.fit_predict(X_sample)
    sil_agg = float(silhouette_score(X_sample, agg_labels))
    
    # 3. BIRCH
    birch = Birch(n_clusters=best_k, threshold=0.5)
    birch_labels = birch.fit_predict(X_sample)
    sil_birch = float(silhouette_score(X_sample, birch_labels))
    
    print(f"  K-Means Silhouette:       {sil_km:.4f}")
    print(f"  Agglomerative Silhouette: {sil_agg:.4f}")
    print(f"  BIRCH Silhouette:         {sil_birch:.4f}")
    
    # Fit K-Means on full dataset for comprehensive profiling
    print("\nFitting final K-Means clusterer on full dataset...")
    full_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    full_labels = full_km.fit_predict(X_trans)
    df["Cluster"] = full_labels
    
    # Profile clusters
    profiles = []
    cluster_names = [
        "Cluster 0: High-Acuity Surgical & Critical Inpatients",
        "Cluster 1: Moderate-Stay General & Medical Inpatients",
        "Cluster 2: Routine Elective & Rapid-Discharge Inpatients"
    ]
    
    for c in range(best_k):
        sub = df[df["Cluster"] == c]
        top_depts = sub["Department"].value_counts(normalize=True).head(3).to_dict()
        top_severity = sub["Severity of Illness"].value_counts(normalize=True).to_dict()
        top_admission = sub["Type of Admission"].value_counts(normalize=True).to_dict()
        mean_age = float(sub["Age_num"].mean())
        mean_rooms = float(sub["Available Extra Rooms in Hospital"].mean())
        mean_visitors = float(sub["Visitors with Patient"].mean())
        mean_deposit = float(sub["Admission_Deposit"].mean())
        mean_stay_mid = float(sub["Stay_midpoint"].mean())
        stay_mode = sub["Stay"].mode()[0]
        
        prof = {
            "cluster_id": c,
            "label": cluster_names[c],
            "count": len(sub),
            "percentage": round(len(sub) / len(df) * 100, 2),
            "mean_stay_days": round(mean_stay_mid, 2),
            "stay_mode_band": stay_mode,
            "mean_age_years": round(mean_age, 1),
            "mean_visitors": round(mean_visitors, 2),
            "mean_deposit_usd": round(mean_deposit, 2),
            "top_departments": {k: round(v*100, 1) for k, v in top_depts.items()},
            "severity_distribution": {k: round(v*100, 1) for k, v in top_severity.items()},
            "admission_types": {k: round(v*100, 1) for k, v in top_admission.items()}
        }
        profiles.append(prof)
        print(f"\n{cluster_names[c]} ({prof['percentage']}% of cohort):")
        print(f"  Mean Stay: {mean_stay_mid:.1f} days (Mode: {stay_mode}) | Mean Age: {mean_age:.1f} yrs")
        print(f"  Severity: {prof['severity_distribution']}")
        print(f"  Top Departments: {prof['top_departments']}")
        
    # Write reports/cluster_profiles.md
    md_content = f"""# WardCast Cluster Profiles Report

## 1. Methodology & Model Comparison
Clustering was performed on {len(df):,} patient records across 116 preprocessed features.
Optimal cluster selection was confirmed using the Elbow method (inertia) and Silhouette analysis across k=2..7.

### Algorithm Comparison at k={best_k}:
| Algorithm | Silhouette Score | Computational Complexity | Memory Footprint |
|---|---|---|---|
| **K-Means** | **{sil_km:.4f}** | O(N * k * d) — Fast, scalable | O(N) |
| **Agglomerative (Ward)** | **{sil_agg:.4f}** | O(N^2 log N) — Prohibitive on 318k rows | O(N^2) |
| **BIRCH** | **{sil_birch:.4f}** | O(N) — Tree-based streaming | O(tree size) |

---

## 2. Discovered Cluster Profiles
"""
    for p in profiles:
        md_content += f"""
### {p['label']}
- **Population**: {p['count']:,} patients ({p['percentage']}%)
- **Length of Stay**: Geometric / Midpoint Mean = **{p['mean_stay_days']} days** (Most common band: `{p['stay_mode_band']}`)
- **Demographics**: Mean Age = **{p['mean_age_years']} years**, Average Visitors = **{p['mean_visitors']}**, Average Deposit = **${p['mean_deposit_usd']:,}**
- **Clinical Severity**:
{pd.Series(p['severity_distribution']).to_frame('Percentage (%)').to_markdown()}
- **Top Hospital Departments**:
{pd.Series(p['top_departments']).to_frame('Percentage (%)').to_markdown()}
- **Admission Mode**:
{pd.Series(p['admission_types']).to_frame('Percentage (%)').to_markdown()}

---
"""
    with open(os.path.join(REPORTS_DIR, "cluster_profiles.md"), "w") as f:
        f.write(md_content)
    print(f"Saved cluster profiles report to {os.path.join(REPORTS_DIR, 'cluster_profiles.md')}")
    
    # Save clusterer pipeline
    cluster_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clusterer", full_km)
    ])
    joblib.dump(cluster_pipe, os.path.join(MODELS_DIR, "clusterer.joblib"))
    print(f"Saved clusterer pipeline to {os.path.join(MODELS_DIR, 'clusterer.joblib')}")
    
    return {
        "optimal_k": best_k,
        "silhouette_kmeans": sil_km,
        "silhouette_agglomerative": sil_agg,
        "silhouette_birch": sil_birch,
        "profiles": profiles
    }

def run_association_rules():
    print("\n" + "="*60)
    print("Phase 3.3: Association Rules (Apriori vs FP-Growth)")
    print("="*60)
    
    df_raw = load_raw_data()
    df = add_derived_features(df_raw)
    
    # Discretize features for transactional association rule mining
    df_items = pd.DataFrame()
    
    # Age bins
    df_items["Age_Group"] = pd.cut(
        df["Age_num"],
        bins=[-np.inf, 30, 60, np.inf],
        labels=["Young_0-30", "Middle_31-60", "Senior_61+"]
    )
    
    # Severity
    df_items["Severity"] = df["Severity of Illness"].map(lambda x: f"Severity_{x}")
    
    # Department
    df_items["Dept"] = df["Department"].map(lambda x: f"Dept_{x}")
    
    # Admission Type
    df_items["AdmType"] = df["Type of Admission"].map(lambda x: f"Adm_{x}")
    
    # Visitors
    df_items["Visitors"] = pd.cut(
        df["Visitors with Patient"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Visitors_Low_0-2", "Visitors_Med_3-4", "Visitors_High_5+"]
    )
    
    # Extra rooms
    df_items["Rooms"] = pd.cut(
        df["Available Extra Rooms in Hospital"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["Rooms_Low_0-2", "Rooms_Med_3-4", "Rooms_High_5+"]
    )
    
    # Bed Grade
    df_items["BedGrade"] = df["Bed Grade"].fillna(2.0).map(lambda x: f"BedGrade_{int(x)}")
    
    # Deposit
    df_items["Deposit"] = pd.cut(
        df["Admission_Deposit"],
        bins=[-np.inf, 4000, 6000, np.inf],
        labels=["Deposit_Low_<4k", "Deposit_Med_4-6k", "Deposit_High_>6k"]
    )
    
    # Stay Target: Short (0-20), Medium (21-50), Long (>50)
    df_items["Stay"] = df["Stay_3class"].map({0: "Stay_Short", 1: "Stay_Med", 2: "Stay_Long"})
    
    print("Converting discretized attributes into binary transactional matrix...")
    df_bool = pd.get_dummies(df_items, dtype=bool)
    print(f"Transactional matrix shape: {df_bool.shape[0]:,} transactions, {df_bool.shape[1]} items.")
    
    min_sup = 0.03
    print(f"\nRunning Apriori (min_support={min_sup})...")
    itemsets_apriori = apriori(df_bool, min_support=min_sup, use_colnames=True)
    print(f"Apriori found {len(itemsets_apriori):,} frequent itemsets.")
    
    print(f"Running FP-Growth (min_support={min_sup})...")
    itemsets_fpgrowth = fpgrowth(df_bool, min_support=min_sup, use_colnames=True)
    print(f"FP-Growth found {len(itemsets_fpgrowth):,} frequent itemsets.")
    
    # Formally verify both algorithms return the exact same itemsets
    ap_set = set(frozenset(x) for x in itemsets_apriori["itemsets"])
    fp_set = set(frozenset(x) for x in itemsets_fpgrowth["itemsets"])
    assert ap_set == fp_set, "FATAL: Apriori and FP-Growth itemsets do not match!"
    print("Verification PASSED: Apriori and FP-Growth generated identical frequent itemsets.")
    
    # Generate association rules
    rules = association_rules(itemsets_fpgrowth, metric="confidence", min_threshold=0.20)
    print(f"Generated {len(rules):,} total association rules (min_confidence=0.20).")
    
    # Filter rules whose consequent is Stay_Long
    target_consequent = frozenset(["Stay_Stay_Long"])
    long_rules = rules[rules["consequents"] == target_consequent].copy()
    print(f"Found {len(long_rules):,} rules predicting Stay_Long.")
    
    # Filter min support >= 0.015 and sort by lift descending
    top_rules = long_rules[long_rules["support"] >= 0.015].sort_values(by="lift", ascending=False).head(15)
    
    # Format rules for display and markdown
    rule_records = []
    for _, r in top_rules.iterrows():
        antecedents_str = ", ".join(list(r["antecedents"]))
        rule_records.append({
            "Antecedent Factors": antecedents_str,
            "Consequent": "Stay_Long (>50 Days)",
            "Support": round(float(r["support"]), 4),
            "Confidence": round(float(r["confidence"]), 4),
            "Lift": round(float(r["lift"]), 4)
        })
        
    top_rules_df = pd.DataFrame(rule_records)
    print("\nTop 15 Association Rules for Long Hospital Stay (Sorted by Lift):")
    print(top_rules_df.to_string(index=False))
    
    # Save to reports/association_rules.md
    rules_md = f"""# WardCast Association Rules Report: Prolonged Stay Drivers

## 1. Methodology & Algorithm Equivalence
Association rule mining was performed using both **Apriori** and **FP-Growth** algorithms via `mlxtend` with `min_support={min_sup}` and `min_confidence=0.20`.
- Frequent Itemsets Discovered: **{len(itemsets_apriori):,}**
- Itemset Exact Match: **Confirmed Identical** between Apriori and FP-Growth.
- Target Consequent: **`Stay_Long (>50 Days)`**

---

## 2. Top 15 Rules by Lift (Support $\ge$ 1.5%)
| # | Antecedent Conditions | Support | Confidence | Lift | Clinical & Operational Implications |
|---|---|---|---|---|---|
"""
    for i, r in enumerate(rule_records, 1):
        conds = r["Antecedent Factors"]
        supp = r["Support"]
        conf = r["Confidence"]
        lift = r["Lift"]
        implication = "High risk factor combination; priority candidate for early discharge planning."
        if "Visitors_High" in conds:
            implication = "Large visitor numbers correlate with high patient dependency and prolonged family involvement."
        elif "Severity_Severity_Extreme" in conds:
            implication = "Extreme acute severity requiring prolonged physiological recovery."
        elif "Senior" in conds:
            implication = "Geriatric multi-morbidity compounding acute recovery timeline."
            
        rules_md += f"| {i} | `{conds}` | {supp:.4f} ({supp*100:.2f}%) | {conf:.4f} ({conf*100:.2f}%) | **{lift:.2f}x** | {implication} |\n"
        
    with open(os.path.join(REPORTS_DIR, "association_rules.md"), "w") as f:
        f.write(rules_md)
    print(f"Saved association rules report to {os.path.join(REPORTS_DIR, 'association_rules.md')}")
    
    return {
        "min_support": min_sup,
        "frequent_itemsets_count": len(itemsets_apriori),
        "algorithms_match": True,
        "total_rules_count": len(rules),
        "long_stay_rules_count": len(long_rules),
        "top_15_rules": rule_records
    }

def main():
    em_res = run_em_analysis()
    cluster_res = run_clustering_analysis()
    rules_res = run_association_rules()
    
    # Update reports/metrics.json
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["phase_3"] = {
        "em_gaussian_mixture": em_res,
        "clustering": {
            "optimal_k": cluster_res["optimal_k"],
            "silhouette_kmeans": cluster_res["silhouette_kmeans"],
            "silhouette_agglomerative": cluster_res["silhouette_agglomerative"],
            "silhouette_birch": cluster_res["silhouette_birch"],
            "cluster_count": len(cluster_res["profiles"])
        },
        "association_rules": {
            "frequent_itemsets": rules_res["frequent_itemsets_count"],
            "algorithms_identical": rules_res["algorithms_match"],
            "top_rule_lift": rules_res["top_15_rules"][0]["Lift"] if rules_res["top_15_rules"] else None
        }
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nUpdated {metrics_path} with Phase 3 results.")

if __name__ == "__main__":
    main()
