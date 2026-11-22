"""
src/prep.py
Phase 1: Preprocessing pipeline, feature dictionary, and modular transforms for Kaggle data.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "Data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = "reports"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

STAY_BANDS = [
    "0-10", "11-20", "21-30", "31-40", "41-50",
    "51-60", "61-70", "71-80", "81-90", "91-100", "More than 100 Days"
]

BAND_TO_LABEL = {band: i for i, band in enumerate(STAY_BANDS)}
LABEL_TO_BAND = {i: band for i, band in enumerate(STAY_BANDS)}

BAND_TO_MIDPOINT = {
    "0-10": 5.0,
    "11-20": 15.0,
    "21-30": 25.0,
    "31-40": 35.0,
    "41-50": 45.0,
    "51-60": 55.0,
    "61-70": 65.0,
    "71-80": 75.0,
    "81-90": 85.0,
    "91-100": 95.0,
    "More than 100 Days": 110.0
}

BAND_TO_3CLASS = {
    "0-10": 0, "11-20": 0,
    "21-30": 1, "31-40": 1, "41-50": 1,
    "51-60": 2, "61-70": 2, "71-80": 2, "81-90": 2, "91-100": 2, "More than 100 Days": 2
}
CLASS_3_NAMES = {0: "Short (0-20d)", 1: "Medium (21-50d)", 2: "Long (>50d)"}

AGE_MAP = {
    "0-10": 5, "11-20": 15, "21-30": 25, "31-40": 35, "41-50": 45,
    "51-60": 55, "61-70": 65, "71-80": 75, "81-90": 85, "91-100": 95
}

SEVERITY_MAP = {
    "Minor": 1,
    "Moderate": 2,
    "Extreme": 3
}

NOMINAL_CATEGORICAL_COLS = [
    "Hospital_code",
    "Hospital_type_code",
    "City_Code_Hospital",
    "Hospital_region_code",
    "Department",
    "Ward_Type",
    "Ward_Facility_Code",
    "City_Code_Patient",
    "Type of Admission"
]

ORDINAL_NUMERIC_COLS = [
    "Age_num",
    "Severity_num",
    "Bed Grade"
]

NUMERIC_BASE_COLS = [
    "Available Extra Rooms in Hospital",
    "Visitors with Patient"
]

def load_raw_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "train_data.csv")
    df = pd.read_csv(filepath)
    return df

def add_derived_features(df):
    """Adds cleanly parsed numeric representations of ordinal fields."""
    df_out = df.copy()
    if "Age" in df_out.columns:
        df_out["Age_num"] = df_out["Age"].map(AGE_MAP)
    if "Severity of Illness" in df_out.columns:
        df_out["Severity_num"] = df_out["Severity of Illness"].map(SEVERITY_MAP)
    if "Stay" in df_out.columns:
        df_out["Stay_label"] = df_out["Stay"].map(BAND_TO_LABEL)
        df_out["Stay_midpoint"] = df_out["Stay"].map(BAND_TO_MIDPOINT)
        df_out["Stay_3class"] = df_out["Stay"].map(BAND_TO_3CLASS)
    return df_out

def get_preprocessor(include_deposit=True, scale=True):
    """
    Returns a scikit-learn ColumnTransformer that fits purely on training data.
    """
    numeric_cols = list(NUMERIC_BASE_COLS)
    if include_deposit:
        numeric_cols.append("Admission_Deposit")
        
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)
    
    ord_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if scale:
        ord_steps.append(("scaler", StandardScaler()))
    ordinal_pipeline = Pipeline(steps=ord_steps)
    
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
    categorical_pipeline = Pipeline(steps=cat_steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ORDINAL_NUMERIC_COLS),
            ("cat", categorical_pipeline, NOMINAL_CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def get_iqr_outlier_mask(df, num_cols=None, factor=1.5):
    """
    Identifies outliers using 1.5x IQR on specified numerical columns.
    Returns boolean Series where True indicates an outlier to be removed.
    """
    if num_cols is None:
        num_cols = ["Available Extra Rooms in Hospital", "Visitors with Patient", "Admission_Deposit"]
    is_outlier = pd.Series(False, index=df.index)
    for col in num_cols:
        if col in df.columns:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            low = q25 - factor * iqr
            high = q75 + factor * iqr
            is_outlier = is_outlier | (df[col] < low) | (df[col] > high)
    return is_outlier

def generate_feature_dictionary(df_clean, preprocessor_with_dep, feature_names_with_dep):
    doc = f"""# WardCast Feature Dictionary

## 1. Overview
The dataset contains patient admission records with clinical, facility, and administrative variables.
All models strictly group cross-validation splits by `patientid` using `GroupKFold(5)` to prevent data leakage.
Total Records: {len(df_clean):,}
Total Raw Features: {df_clean.shape[1] - 1} (excluding target)
Total Preprocessed Features (with Admission_Deposit): {len(feature_names_with_dep)}

---

## 2. Target Variables
| Target Name | Type | Categories / Domain | Description |
|---|---|---|---|
| `Stay` | Categorical Band | 11 ordinal intervals (`0-10` to `More than 100 Days`) | Raw length of stay band. |
| `Stay_label` | Integer Ordinal | 0 to 10 | Ordinal index mapped from `Stay`. Primary classification target. |
| `Stay_3class` | Categorical 3-class | 0: Short (0-20d, 31.9%), 1: Medium (21-50d, 48.5%), 2: Long (>50d, 19.6%) | Aggregated 3-tier hospital operational target. |
| `Stay_midpoint` | Continuous (Days) | 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 110.0 | Target for multivariable linear regression and LASSO. |

---

## 3. Raw Predictor Variables
| Column Name | Raw Type | Preprocessing Action | Imputation Rule | Domain / Values |
|---|---|---|---|---|
| `Hospital_code` | int64 | Categorical (OneHotEncoder) | Mode | 32 hospital facilities. |
| `Hospital_type_code` | str | Categorical (OneHotEncoder) | Mode | Types `a` through `g`. |
| `City_Code_Hospital` | int64 | Categorical (OneHotEncoder) | Mode | 11 city codes. |
| `Hospital_region_code` | str | Categorical (OneHotEncoder) | Mode | Regions `X`, `Y`, `Z`. |
| `Available Extra Rooms in Hospital` | int64 | Numeric (StandardScaler) | Median | Integer rooms count (0 to 24). |
| `Department` | str | Categorical (OneHotEncoder) | Mode | radiotherapy, anesthesia, gynecology, TB & Chest disease, surgery. |
| `Ward_Type` | str | Categorical (OneHotEncoder) | Mode | Wards `P`, `Q`, `R`, `S`, `T`, `U`. |
| `Ward_Facility_Code` | str | Categorical (OneHotEncoder) | Mode | Facilities `A` through `F`. |
| `Bed Grade` | float64 | Ordinal Numeric (StandardScaler) | Mode (0.04% missing) | Grades 1.0, 2.0, 3.0, 4.0. |
| `City_Code_Patient` | float64 | Categorical (OneHotEncoder) | Mode (1.42% missing) | 37 distinct patient geographic city codes. |
| `Type of Admission` | str | Categorical (OneHotEncoder) | Mode | Emergency, Trauma, Urgent. |
| `Severity of Illness` | str | Ordinal Mapped (`Severity_num`: 1=Minor, 2=Moderate, 3=Extreme) | Mode | Minor, Moderate, Extreme. |
| `Visitors with Patient` | int64 | Numeric (StandardScaler) | Median | Accompanying visitors (0 to 32). |
| `Age` | str | Ordinal Mapped (`Age_num`: midpoints 5 to 95) | Mode | Age decades `0-10` through `91-100`. |
| `Admission_Deposit` | float64 | Numeric (StandardScaler) | Median | Currency deposit paid at admission ($1,800 to $11,000). Controlled via ablation toggle. |

---

## 4. Identifiers Excluded From Model Features
- `case_id`: Unique record row ID. Excluded to prevent spurious correlation.
- `patientid`: Unique patient ID. Excluded from features; used exclusively as the grouping key in `GroupKFold`.

---

## 5. Outlier Detection Strategy
- Evaluated on numerical columns (`Available Extra Rooms in Hospital`, `Visitors with Patient`, `Admission_Deposit`) using 1.5x IQR.
- Computed solely on training folds during cross-validation to maintain strict zero-leakage protocol.
"""
    with open(os.path.join(REPORTS_DIR, "feature_dictionary.md"), "w") as f:
        f.write(doc)

def main():
    print("Loading raw training data...")
    df = load_raw_data()
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    
    df_derived = add_derived_features(df)
    
    # Check preprocessors
    prep_with_dep = get_preprocessor(include_deposit=True, scale=True)
    prep_no_dep = get_preprocessor(include_deposit=False, scale=True)
    
    X_sample = df_derived.drop(columns=["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class"])
    prep_with_dep.fit(X_sample)
    
    feature_names = prep_with_dep.get_feature_names_out()
    print(f"Preprocessed feature count (with deposit): {len(feature_names)}")
    
    prep_no_dep.fit(X_sample.drop(columns=["Admission_Deposit"]))
    feature_names_no_dep = prep_no_dep.get_feature_names_out()
    print(f"Preprocessed feature count (without deposit): {len(feature_names_no_dep)}")
    
    # Generate feature dictionary
    generate_feature_dictionary(df_derived, prep_with_dep, feature_names)
    print("Saved reports/feature_dictionary.md")
    
    # Save metadata
    metadata = {
        "stay_bands": STAY_BANDS,
        "band_to_label": BAND_TO_LABEL,
        "band_to_midpoint": BAND_TO_MIDPOINT,
        "band_to_3class": BAND_TO_3CLASS,
        "class_3_names": CLASS_3_NAMES,
        "age_map": AGE_MAP,
        "severity_map": SEVERITY_MAP,
        "nominal_categorical_cols": NOMINAL_CATEGORICAL_COLS,
        "ordinal_numeric_cols": ORDINAL_NUMERIC_COLS,
        "numeric_base_cols": NUMERIC_BASE_COLS,
        "feature_count_with_deposit": len(feature_names),
        "feature_count_without_deposit": len(feature_names_no_dep),
        "feature_names_with_deposit": list(feature_names),
        "feature_names_without_deposit": list(feature_names_no_dep),
    }
    
    metadata_path = os.path.join(PROCESSED_DIR, "prep_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved preprocessing metadata to {metadata_path}")

if __name__ == "__main__":
    main()
