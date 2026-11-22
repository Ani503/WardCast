# WardCast Feature Dictionary

## 1. Overview
The dataset contains patient admission records with clinical, facility, and administrative variables.
All models strictly group cross-validation splits by `patientid` using `GroupKFold(5)` to prevent data leakage.
Total Records: 318,438
Total Raw Features: 22 (excluding target)
Total Preprocessed Features (with Admission_Deposit): 116

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
