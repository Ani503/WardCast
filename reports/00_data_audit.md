# WardCast Phase 0: Data Audit Report

## 1. Executive Summary & Explicit Answers to Core Audit Questions

### (a) Is Kaggle `Stay` in bands or exact days?
**Answer: Bands.** 
The `Stay` target column in `train_data.csv` consists strictly of 11 discrete ordinal stay duration bands:
`0-10` (23,604 rows, 7.41%), `11-20` (78,139 rows, 24.54%), `21-30` (87,491 rows, 27.48%), `31-40` (55,159 rows, 17.32%), `41-50` (11,743 rows, 3.69%), `51-60` (35,018 rows, 11.0%), `61-70` (2,744 rows, 0.86%), `71-80` (10,254 rows, 3.22%), `81-90` (4,838 rows, 1.52%), `91-100` (2,765 rows, 0.87%), `More than 100 Days` (6,683 rows, 2.1%).
There is no exact day count in the Kaggle dataset.

### (b) Does SPARCS contain exact admission or discharge dates?
**Answer: No exact dates.**
The SPARCS dataset (`Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv`) contains `['Discharge Year']` (specifically `Discharge Year` = 2014). It does **not** contain exact admission dates, discharge dates, or timestamps to preserve patient de-identification. However, unlike Kaggle, its length of stay column (`Length of Stay`) records continuous numerical stay days, capped at `120 +`.

### (c) Which columns in HHS give beds and beds used, and what is the date range and number of states?
**Answer:**
- Total Inpatient Beds: `inpatient_beds` (available across facilities/state)
- Total Inpatient Beds Used: `inpatient_beds_used`
- Date Range: `2020-01-01` to `2024-04-20`
- Total Geographic Units: `54` states/territories/districts.

---

## 2. Dataset Deep Dives

### 2.1 Kaggle Healthcare Analytics II (`train_data.csv` & `test_data.csv`)
- **Dimensions**: 318,438 rows, 18 columns.
- **Duplicates**: 0 duplicate rows.
- **Target (`Stay`)**: 11 bands. Class distribution is imbalanced:
|                    |   Count |   Percentage (%) |
|:-------------------|--------:|-----------------:|
| 21-30              |   87491 |            27.48 |
| 11-20              |   78139 |            24.54 |
| 31-40              |   55159 |            17.32 |
| 51-60              |   35018 |            11    |
| 0-10               |   23604 |             7.41 |
| 41-50              |   11743 |             3.69 |
| 71-80              |   10254 |             3.22 |
| More than 100 Days |    6683 |             2.1  |
| 81-90              |    4838 |             1.52 |
| 91-100             |    2765 |             0.87 |
| 61-70              |    2744 |             0.86 |
- **Missing Values**:
  - `City_Code_Patient`: 4,532 (1.42%)
  - `Bed Grade`: 113 (0.04%)
  - All other columns have 0 missing values.
- **Patient Grouping (`patientid`)**:
  - Unique patients: 92,017
  - Patients with multiple admissions: 71,668 (77.89%)
  - Max admissions for a single patient: 50
  - Mean admissions per patient: 3.46
  - **CRITICAL IMPLICATION**: Standard K-Fold CV would cause data leakage across admissions for the same patient. **`GroupKFold(n_splits=5)` grouped strictly by `patientid` must be used for all CV splits.**
- **Test File (`test_data.csv`)**:
  - Dimensions: 137,057 rows, 17 columns.
  - Has `Stay` column: **False**. As expected, `test_data.csv` is completely unlabelled. Per brief instructions, it will not be used for model scoring or evaluation.

### 2.2 NY SPARCS Inpatient Discharges 2014 (`Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2014_20261121.csv`)
- **Dimensions**: 2,367,550 rows, 34 columns.
- **Length of Stay Column**: `Length of Stay`. Sample values: `['8', '3', '1', '5', '4', '7', '2', '6', '18', '20']`. Note strings like `'120 +'` requiring parsing to integer.
- **Date Information**: De-identified annual record for 2014 (`Discharge Year`). No calendar days or admission dates present.
- **Hospital Identifier**: `Facility Name` / `Facility Id` allows hospital-held-out cross-validation for Phase 4.

### 2.3 HHS Hospital Capacity Time Series (`COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv`)
- **Dimensions**: 81,335 rows, 135 columns.
- **Date Range**: 2020-01-01 through 2024-04-20.
- **States & Territories**: 54 (covers all 50 states, DC, PR, VI, etc.).
- **Key Capacity Columns**:
  - `inpatient_beds`: Total number of reported inpatient beds. Missing values: 351.
  - `inpatient_beds_used`: Total number of reported inpatient beds occupied. Missing values: 351.
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
