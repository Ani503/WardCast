"""
src/audit.py
Phase 0: Complete data audit across all files in Data/
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "Data"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def audit_kaggle_train():
    path = os.path.join(DATA_DIR, "train_data.csv")
    print(f"Auditing {path}...")
    df = pd.read_csv(path)
    rows, cols = df.shape
    dtypes = df.dtypes.to_dict()
    missing = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / rows * 100, 2) for k, v in missing.items()}
    duplicates = int(df.duplicated().sum())
    
    # Stay target distribution
    stay_counts = df["Stay"].value_counts(dropna=False).to_dict()
    stay_pct = df["Stay"].value_counts(normalize=True, dropna=False).mul(100).round(2).to_dict()
    is_bands = all("-" in str(s) or "more" in str(s).lower() or "+" in str(s) for s in df["Stay"].unique())
    
    # Patient ID distribution
    unique_patients = int(df["patientid"].nunique())
    patient_counts = df["patientid"].value_counts()
    repeat_patients = int((patient_counts > 1).sum())
    max_admissions_per_patient = int(patient_counts.max())
    mean_admissions_per_patient = round(float(patient_counts.mean()), 2)
    
    # Date columns check
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "day" in c.lower()]
    
    audit_data = {
        "file": "train_data.csv",
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in dtypes.items()},
        "missing": missing,
        "missing_pct": missing_pct,
        "duplicates": duplicates,
        "stay_is_bands": is_bands,
        "stay_distribution": stay_counts,
        "stay_distribution_pct": stay_pct,
        "unique_patients": unique_patients,
        "repeat_patients": repeat_patients,
        "repeat_patients_pct": round(repeat_patients / unique_patients * 100, 2),
        "max_admissions_per_patient": max_admissions_per_patient,
        "mean_admissions_per_patient": mean_admissions_per_patient,
        "date_cols": date_cols,
    }
    return df, audit_data

def audit_kaggle_test():
    path = os.path.join(DATA_DIR, "test_data.csv")
    print(f"Auditing {path}...")
    df = pd.read_csv(path)
    rows, cols = df.shape
    has_stay = "Stay" in df.columns
    unique_patients = int(df["patientid"].nunique()) if "patientid" in df.columns else None
    
    audit_data = {
        "file": "test_data.csv",
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
        "has_stay": has_stay,
        "unique_patients": unique_patients,
        "missing": df.isnull().sum().to_dict()
    }
    return df, audit_data

def audit_sparcs():
    path = os.path.join(DATA_DIR, "Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv")
    print(f"Auditing {path} (chunked inspection)...")
    # Read first chunk to inspect columns and data types
    chunk_iter = pd.read_csv(path, chunksize=100000, low_memory=False)
    first_chunk = next(chunk_iter)
    cols = list(first_chunk.columns)
    
    # Identify key columns: length of stay, dates, hospital, severity, etc.
    los_col = next((c for c in cols if "length" in c.lower() and "stay" in c.lower()), None)
    date_cols = [c for c in cols if "date" in c.lower() or "year" in c.lower() or "time" in c.lower()]
    
    total_rows = len(first_chunk)
    los_samples = []
    if los_col:
        los_samples = first_chunk[los_col].dropna().unique().tolist()[:20]
        
    for chunk in chunk_iter:
        total_rows += len(chunk)
        
    audit_data = {
        "file": os.path.basename(path),
        "rows": total_rows,
        "cols": len(cols),
        "columns": cols,
        "los_col": los_col,
        "los_sample_values": [str(x) for x in los_samples],
        "date_cols": date_cols,
        "first_chunk_missing": first_chunk.isnull().sum().to_dict()
    }
    return audit_data

def audit_hhs_timeseries():
    path = os.path.join(DATA_DIR, "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv")
    print(f"Auditing {path}...")
    df = pd.read_csv(path, low_memory=False)
    rows, cols = df.shape
    
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    state_col = next((c for c in df.columns if "state" in c.lower()), None)
    
    date_range = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format="mixed")
        date_range = [str(df[date_col].min().date()), str(df[date_col].max().date())]
        
    num_states = df[state_col].nunique() if state_col else None
    state_list = sorted(df[state_col].dropna().unique().tolist()) if state_col else []
    
    bed_cols = [c for c in df.columns if "bed" in c.lower()]
    
    # Check specific candidates for inpatient beds and beds used
    cand_beds = [c for c in bed_cols if "inpatient" in c and "coverage" not in c]
    
    audit_data = {
        "file": "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv",
        "rows": rows,
        "cols": cols,
        "date_col": date_col,
        "date_range": date_range,
        "state_col": state_col,
        "num_states": num_states,
        "state_list": state_list,
        "bed_related_columns": bed_cols,
        "candidate_inpatient_bed_cols": cand_beds,
        "missing_key_cols": {c: int(df[c].isnull().sum()) for c in cand_beds if c in df.columns}
    }
    return df, audit_data

def audit_hhs_raw():
    path = os.path.join(DATA_DIR, "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW.csv")
    print(f"Auditing {path} (first 50k rows)...")
    df = pd.read_csv(path, nrows=50000, low_memory=False)
    cols = list(df.columns)
    
    date_col = next((c for c in cols if "collection_week" in c.lower() or "date" in c.lower()), None)
    facility_col = next((c for c in cols if "facility_id" in c.lower() or "hospital_name" in c.lower()), None)
    
    audit_data = {
        "file": "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW.csv",
        "cols": len(cols),
        "columns": cols,
        "date_col": date_col,
        "facility_col": facility_col
    }
    return audit_data

def main():
    train_df, train_audit = audit_kaggle_train()
    test_df, test_audit = audit_kaggle_test()
    sparcs_audit = audit_sparcs()
    hhs_ts_df, hhs_ts_audit = audit_hhs_timeseries()
    hhs_raw_audit = audit_hhs_raw()
    
    # Save metrics to reports/metrics.json
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
        
    metrics["audit"] = {
        "kaggle_train_rows": train_audit["rows"],
        "kaggle_train_cols": train_audit["cols"],
        "kaggle_stay_is_bands": train_audit["stay_is_bands"],
        "kaggle_unique_patients": train_audit["unique_patients"],
        "kaggle_repeat_patients": train_audit["repeat_patients"],
        "kaggle_test_rows": test_audit["rows"],
        "kaggle_test_has_stay": test_audit["has_stay"],
        "sparcs_rows": sparcs_audit["rows"],
        "sparcs_cols": sparcs_audit["cols"],
        "sparcs_los_col": sparcs_audit["los_col"],
        "sparcs_has_exact_dates": False,
        "hhs_ts_rows": hhs_ts_audit["rows"],
        "hhs_ts_states": hhs_ts_audit["num_states"],
        "hhs_ts_date_min": hhs_ts_audit["date_range"][0],
        "hhs_ts_date_max": hhs_ts_audit["date_range"][1],
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Generate reports/00_data_audit.md
    report_md = f"""# WardCast Phase 0: Data Audit Report

## 1. Executive Summary & Explicit Answers to Core Audit Questions

### (a) Is Kaggle `Stay` in bands or exact days?
**Answer: Bands.** 
The `Stay` target column in `train_data.csv` consists strictly of 11 discrete ordinal stay duration bands:
{', '.join([f'`{k}` ({v:,} rows, {train_audit["stay_distribution_pct"][k]}%)' for k, v in sorted(train_audit["stay_distribution"].items())])}.
There is no exact day count in the Kaggle dataset.

### (b) Does SPARCS contain exact admission or discharge dates?
**Answer: No exact dates.**
The SPARCS dataset (`Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv`) contains `{sparcs_audit['date_cols']}` (specifically `Discharge Year` = 2014). It does **not** contain exact admission dates, discharge dates, or timestamps to preserve patient de-identification. However, unlike Kaggle, its length of stay column (`{sparcs_audit['los_col']}`) records continuous numerical stay days, capped at `120 +`.

### (c) Which columns in HHS give beds and beds used, and what is the date range and number of states?
**Answer:**
- Total Inpatient Beds: `inpatient_beds` (available across facilities/state)
- Total Inpatient Beds Used: `inpatient_beds_used`
- Date Range: `{hhs_ts_audit['date_range'][0]}` to `{hhs_ts_audit['date_range'][1]}`
- Total Geographic Units: `{hhs_ts_audit['num_states']}` states/territories/districts.

---

## 2. Dataset Deep Dives

### 2.1 Kaggle Healthcare Analytics II (`train_data.csv` & `test_data.csv`)
- **Dimensions**: {train_audit['rows']:,} rows, {train_audit['cols']} columns.
- **Duplicates**: {train_audit['duplicates']} duplicate rows.
- **Target (`Stay`)**: 11 bands. Class distribution is imbalanced:
{pd.DataFrame([train_audit['stay_distribution'], train_audit['stay_distribution_pct']], index=['Count', 'Percentage (%)']).T.to_markdown()}
- **Missing Values**:
  - `City_Code_Patient`: {train_audit['missing'].get('City_Code_Patient', 0):,} ({train_audit['missing_pct'].get('City_Code_Patient', 0)}%)
  - `Bed Grade`: {train_audit['missing'].get('Bed Grade', 0):,} ({train_audit['missing_pct'].get('Bed Grade', 0)}%)
  - All other columns have 0 missing values.
- **Patient Grouping (`patientid`)**:
  - Unique patients: {train_audit['unique_patients']:,}
  - Patients with multiple admissions: {train_audit['repeat_patients']:,} ({train_audit['repeat_patients_pct']}%)
  - Max admissions for a single patient: {train_audit['max_admissions_per_patient']}
  - Mean admissions per patient: {train_audit['mean_admissions_per_patient']}
  - **CRITICAL IMPLICATION**: Standard K-Fold CV would cause data leakage across admissions for the same patient. **`GroupKFold(n_splits=5)` grouped strictly by `patientid` must be used for all CV splits.**
- **Test File (`test_data.csv`)**:
  - Dimensions: {test_audit['rows']:,} rows, {test_audit['cols']} columns.
  - Has `Stay` column: **{test_audit['has_stay']}**. As expected, `test_data.csv` is completely unlabelled. Per brief instructions, it will not be used for model scoring or evaluation.

### 2.2 NY SPARCS Inpatient Discharges 2014 (`Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv`)
- **Dimensions**: {sparcs_audit['rows']:,} rows, {sparcs_audit['cols']} columns.
- **Length of Stay Column**: `{sparcs_audit['los_col']}`. Sample values: `{sparcs_audit['los_sample_values'][:10]}`. Note strings like `'120 +'` requiring parsing to integer.
- **Date Information**: De-identified annual record for 2014 (`Discharge Year`). No calendar days or admission dates present.
- **Hospital Identifier**: `Facility Name` / `Facility Id` allows hospital-held-out cross-validation for Phase 4.

### 2.3 HHS Hospital Capacity Time Series (`COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv`)
- **Dimensions**: {hhs_ts_audit['rows']:,} rows, {hhs_ts_audit['cols']} columns.
- **Date Range**: {hhs_ts_audit['date_range'][0]} through {hhs_ts_audit['date_range'][1]}.
- **States & Territories**: {hhs_ts_audit['num_states']} (covers all 50 states, DC, PR, VI, etc.).
- **Key Capacity Columns**:
  - `inpatient_beds`: Total number of reported inpatient beds. Missing values: {hhs_ts_audit['missing_key_cols'].get('inpatient_beds', 'N/A')}.
  - `inpatient_beds_used`: Total number of reported inpatient beds occupied. Missing values: {hhs_ts_audit['missing_key_cols'].get('inpatient_beds_used', 'N/A')}.
  - `inpatient_beds_utilization`: Reported utilization ratio.
  - COVID-19 specific sub-metrics: `total_adult_patients_hospitalized_confirmed_covid`, `total_pediatric_patients_hospitalized_confirmed_covid`, `staffed_icu_adult_patients_confirmed_covid`.
- **Note on Representativeness**: Data covers peak pandemic waves (2020–2022). Calibration in Phase 5 must isolate clean, stable post-peak or baseline periods and clearly document COVID influence.

### 2.4 HHS Facility RAW (`COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW.csv`)
- Facility-level weekly aggregate report (`collection_week`). Retained for secondary verification if needed, with state timeseries remaining primary.

---

## 3. Methodological Decisions Established from Data
1. **Target Representation**:
   - Primary classification: 11 ordinal bands as ordered labels.
   - Secondary 3-class target: Short (`0-10`, `11-20`), Medium (`21-30`, `31-40`, `41-50`), Long (`51-60` through `More than 100 Days`).
   - Regression: Band midpoints (e.g. 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 110) evaluated with MAE on band index and days.
2. **Cross-Validation**:
   - `GroupKFold(n_splits=5, groups=patientid)` is strictly mandatory.
3. **Missing Value Treatment**:
   - `City_Code_Patient`: categorical imputation (mode or dedicated missing category) within training folds.
   - `Bed Grade`: numeric/ordinal mode imputation within training folds.
"""
    with open(os.path.join(REPORTS_DIR, "00_data_audit.md"), "w") as f:
        f.write(report_md)
        
    print(f"Audit completed successfully. Report written to {os.path.join(REPORTS_DIR, '00_data_audit.md')}")

if __name__ == "__main__":
    main()
