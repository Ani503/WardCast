# WardCast Backend-to-UI API Specification & Data Contract

**Specification Version**: 1.0.0 (Production Core)  
**Academic Target**: University Course PCCAIML 502  
**Interface Module**: `src/predict.py`  
**Scope Notice**: This document establishes the strict contract for all data exchanges between the WardCast Machine Learning backend and any downstream user interface (UI) or clinical integration.

---

## 1. System-Wide Architectural Standards & Policies

### 1.1 Canonical Public API Endpoints
The WardCast ML interface strictly exposes 6 canonical public API functions:
1. `predict_stay(patient: dict) -> dict`
2. `forecast_7day(current_occupancy: float, beds: int = 120, recent_history: list = None) -> dict`
3. `surge_state(occupancy_history: list[float]) -> dict`
4. `run_scenario(name: str = "flu_season_rush", capacity: int = 120, multiplier: float = 1.35) -> dict`
5. `predict_ward(patients: list[dict], capacity: int = None) -> dict`
6. `get_evidence() -> dict`

### 1.2 Evidence Level Categorization
Every endpoint and operational output in WardCast is tagged with an immutable `evidence_level` field to provide full clinical transparency:
- **`"validated"`**: Statistically robust methods evaluated on multi-fold cross-validation or empirical benchmarks with acceptable clinical error profiles.
  - Inpatient Stay Probability Distributions (`predict_stay`)
  - Long-Stay Alert Trigger at $\theta = 0.25$ (`is_long_stay_alert`)
  - Ward-Level Aggregate Bed-Day Forecasts (`predict_ward`)
  - Calibrated 85% Occupancy Threshold Warning Rules (`rule_warning_level` in `surge_state`)
  - Headline Validation Metrics (`get_evidence`)
- **`"experimental"`**: Exploratory, academic, or queueing simulation components subject to structural domain constraints.
  - Gaussian Hidden Markov Model Regime Decoding (`hmm_warning_level` in `surge_state`)
  - Patient-Level 7-Day Rolling Monte Carlo Census Forecasting (`forecast_7day`)
  - Discrete-Event Epidemic Scenario Simulations (`run_scenario`)

### 1.3 Physical Occupancy Capping & Unmet Demand Semantics
To prevent confusing clinicians and hospital administrators with physically impossible bed allocations:
1. **Physical Bed Occupancy (`occupancy_rate`, `latest_occupancy_rate`, `peak_occupancy`) is STRICTLY capped at `1.0` (100.0%).** A 120-bed hospital cannot physically occupy 136 licensed beds.
2. **Excess Patient Demand is ALWAYS reported separately via `unmet_demand_beds` (integer beds).**
   - E.g., if total incoming demand reaches 136 patients for 120 physical beds:
     - `occupancy_rate`: `1.0000` (100.0% of licensed beds occupied)
     - `unmet_demand_beds`: `16` (16 overflow patients requiring auxiliary cots or diversion)
     - `demand_occupancy_rate` / `peak_occupancy_demand`: `1.1333` (113.3% total patient demand)

### 1.4 Probability Simplex Invariant
All probability vectors (`band_probabilities`, `class_3_probabilities`, `current_state_posteriors`) strictly satisfy the simplex equality:
$$\sum_{k} P_k = 1.0000 \quad \text{and} \quad P_k \ge 0 \quad \forall k$$

---

## 2. API Endpoint Contracts

### 2.1 Endpoint 1: `predict_stay(patient: dict) -> dict`
Predicts calibrated length-of-stay probabilities, operational 3-class triage tier, clinical phenotype cluster, and driving risk factors for an individual inpatient.

- **Evidence Level**: `"validated"`
- **Factor Computation & Signal Attribution**: `top_factors` is computed dynamically per individual patient by evaluating the patient's specific input values (Severity of Illness, Visitors with Patient, Admission Type, Department, and Admission Deposit) against empirical population baselines, rather than returning a static fixed lookup table. Directional signals are reported strictly with non-causal association phrasing (`associated with longer stays (+)` or `associated with shorter stays (-)`), without speculative clinical rationale text.
- **Input Parameters (`patient: dict`)**:

| Parameter Key | Type | Required? | Valid Values / Range | Default (Fallback) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Department` | string | Optional | `'gynecology'`, `'radiotherapy'`, `'anesthesia'`, `'surgery'`, `'TB & Chest disease'` | `'gynecology'` | Clinical service department |
| `Severity of Illness` | string | Optional | `'Minor'`, `'Moderate'`, `'Extreme'` | `'Moderate'` | Clinician-assessed physiological acuity |
| `Age` | string | Optional | `'0-10'`, `'11-20'`, `'21-30'`, `'31-40'`, `'41-50'`, `'51-60'`, `'61-70'`, `'71-80'`, `'81-90'`, `'91-100'` | `'41-50'` | Inpatient age decile |
| `Visitors with Patient`| integer| Optional | $[0, 32]$ (Integer count) | `2` | Number of registered family visitors |
| `Admission_Deposit` | float | Optional | $[1000.0, 15000.0]$ ($USD) | `4800.0` | Initial administrative admission deposit |
| `Type of Admission` | string | Optional | `'Emergency'`, `'Trauma'`, `'Urgent'` | `'Emergency'` | Mode of hospital intake |
| `Bed Grade` | float | Optional | `1.0`, `2.0`, `3.0`, `4.0` | `2.0` | Inpatient bed tier / ward infrastructure |
| `Available Extra Rooms`| integer| Optional | $[0, 24]$ | `3` | Vacant room buffer at intake |

- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'validated'` | Scientific rigor tier |
| `predicted_band` | string | One of 11 stay bands | Primary predicted ordinal stay duration band |
| `predicted_days_midpoint` | float | $5.0 \le y \le 110.0$ | Deterministic midpoint of predicted band (days) |
| `band_probabilities` | dict | Dict of 11 floats summing to 1.0 | Full out-of-fold probability distribution across all 11 bands |
| `class_3_label` | string | `'Short (0-20d)'`, `'Medium (21-50d)'`, `'Long (>50d)'` | Operational 3-class triage tier |
| `class_3_probabilities` | dict | Dict of 3 floats summing to 1.0 | Aggregated probabilities across the 3 operational tiers |
| `is_long_stay_alert` | boolean| `true` or `false` | Early warning flag: triggered if $P(\text{Stay} > 50\text{d}) \ge 0.25$ |
| `long_stay_probability` | float | $0.0 \le p \le 1.0$ | Posterior probability of extended stay (>50 days) |
| `cluster_id` | integer| `0`, `1`, `2` | Patient phenotype cluster identifier |
| `cluster_label` | string | `'Cluster A'`, `'Cluster B'`, `'Cluster C'` | Descriptive cluster name |
| `top_3_factors` | list[str]| 3 text strings | Top driving clinical/administrative factors |
| `top_factors` | list[dict]| List of dicts (`factor`, `impact`) | Dynamic per-patient empirical associations |

- **Example JSON Request**:
```json
{
  "Department": "radiotherapy",
  "Severity of Illness": "Extreme",
  "Age": "51-60",
  "Visitors with Patient": 6,
  "Admission_Deposit": 6200.0,
  "Type of Admission": "Emergency",
  "Bed Grade": 3.0
}
```

- **Example JSON Response**:
```json
{
  "status": "success",
  "evidence_level": "validated",
  "top_band": "51-60",
  "predicted_band": "51-60",
  "predicted_days_midpoint": 55.0,
  "band_probabilities": {
    "0-10": 0.0003,
    "11-20": 0.0592,
    "21-30": 0.0141,
    "31-40": 0.3391,
    "41-50": 0.0124,
    "51-60": 0.3821,
    "61-70": 0.0071,
    "71-80": 0.0982,
    "81-90": 0.0224,
    "91-100": 0.0251,
    "More than 100 Days": 0.0400
  },
  "class_3_label": "Long (>50d)",
  "class_3_probabilities": {
    "Short (0-20d)": 0.0595,
    "Medium (21-50d)": 0.3656,
    "Long (>50d)": 0.5749
  },
  "is_long_stay_alert": true,
  "long_stay_probability": 0.5749,
  "cluster_id": 0,
  "cluster_label": "Cluster A",
  "top_3_factors": [
    "Illness Severity: Extreme",
    "High Visitor Count (6 visitors)",
    "Department: Radiotherapy"
  ],
  "top_factors": [
    {
      "factor": "Illness Severity: Extreme",
      "impact": "associated with longer stays (+)"
    },
    {
      "factor": "High Visitor Count (6 visitors)",
      "impact": "associated with longer stays (+)"
    },
    {
      "factor": "Department: Radiotherapy",
      "impact": "associated with longer stays (+)"
    }
  ]
}
```

---

### 2.2 Endpoint 2: `forecast_7day(current_occupancy: float, beds: int = 120, recent_history: list = None) -> dict`
Generates a 7-day rolling inpatient bed census trajectory. If `recent_history` is omitted or empty, returns the empirical persistence forecast (`forecast_basis: 'persistence'`). If `recent_history` is provided with trailing daily occupancy rates, executes discrete-event Monte Carlo simulation (`forecast_basis: 'simulation'`).

- **Evidence Level**: `"experimental"`
- **Input Parameters**:
  - `current_occupancy` (float): Current hospital occupancy rate ($0.0 \le O \le 1.0$) or active census count ($N > 1.0$).
  - `beds` (integer, default `120`): Licensed physical bed capacity.
  - `recent_history` (list[float], optional): Trailing daily occupancy rates. If omitted or empty, returns persistence baseline.
- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'experimental'` | Scientific rigor tier |
| `forecast_basis` | string | `'persistence'` or `'simulation'` | Forecasting mechanism utilized |
| `empirical_coverage` | float | `0.733` | Empirical 80% CI coverage on held-out test window |
| `is_simulated` | boolean| `true` | Simulation flag |
| `average_stay_days` | float | `32.3` | Training stay distribution mean length of stay (days) |
| `capacity` | integer| Physical bed count | Configured licensed bed capacity |
| `current_occupancy` | float | $0.0 \le O \le 1.0$ | Physical bed occupancy rate (capped at 1.0) |
| `current_occupancy_demand`| float | $O \ge 0.0$ | Total patient demand rate |
| `current_unmet_demand_beds`| integer| $\ge 0$ | Unmet bed demand count |
| `daily_occupancy` | list[float]| 7 floats in $[0.0, 1.0]$ | Physical bed occupancy per day (Days +1 to +7) |
| `daily_census` | list[float]| 7 floats $\ge 0.0$ | Median expected inpatient bed count |
| `surge_risk_per_day` | list[float]| 7 floats in $[0.0, 1.0]$ | Probability of census breaching $\ge 90\%$ capacity |
| `unmet_demand_beds` | list[float]| 7 floats $\ge 0.0$ | Unmet patient bed demand per day |
| `prediction_intervals` | list[dict] | 7 objects (`range_low`, `range_high`) | 80% empirical prediction interval bounds |
| `warning_level` | string | `'NORMAL'`, `'SURGE ALERT (PRE-SURGE)'`, `'CRITICAL SURGE'` | Operational warning tier |
| `days` | list[dict] | 7 detailed horizon objects | Full horizon breakdown |

- **Example JSON Request**:
```json
{
  "current_occupancy": 0.78,
  "beds": 120
}
```

- **Example JSON Response (Summary)**:
```json
{
  "status": "success",
  "evidence_level": "experimental",
  "forecast_basis": "persistence",
  "empirical_coverage": 0.733,
  "is_simulated": true,
  "average_stay_days": 32.3,
  "capacity": 120,
  "current_occupancy": 0.7800,
  "current_occupancy_demand": 0.7800,
  "current_unmet_demand_beds": 0,
  "daily_occupancy": [0.7583, 0.7333, 0.7167, 0.7000, 0.6917, 0.6750, 0.6583],
  "daily_census": [91.0, 88.0, 86.0, 84.0, 83.0, 81.0, 79.0],
  "surge_risk_per_day": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "unmet_demand_beds": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "shortfall_beds_per_day": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "prediction_intervals": [
    {"range_low": 87.0, "range_high": 94.1},
    {"range_low": 84.0, "range_high": 93.0},
    {"range_low": 80.0, "range_high": 92.0},
    {"range_low": 78.0, "range_high": 91.0},
    {"range_low": 76.0, "range_high": 90.0},
    {"range_low": 74.0, "range_high": 89.0},
    {"range_low": 72.0, "range_high": 88.0}
  ],
  "warning_level": "NORMAL"
}
```

---

### 2.3 Endpoint 3: `surge_state(occupancy_history: list[float]) -> dict`
Decodes the current latent operational regime using the 4-state Gaussian Hidden Markov Model and evaluates 1-day ahead physical surge risk ($\ge 90\%$). Strictly isolates the rule-based warning tier from the experimental HMM inference.

- **Evidence Level**: `"experimental"` (HMM State Decoding) / `"validated"` (Calibrated 85% Rule Warning)
- **Input Parameters (`occupancy_history: list[float]`)**:
  - Chronologically ordered list of daily occupancy rates ($0.0 \le O_t \le 1.0$), with at least 2 historical days.
- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'experimental'` | Model architectural rigor tier |
| `rule_evidence_level` | string | `'validated'` | Calibrated threshold rule tier |
| `warning_rule_evidence_level`| string | `'validated'` | Calibrated threshold rule tier |
| `hmm_evidence_level` | string | `'experimental'` | Latent state model tier |
| `state_label` | string | One of 4 state names | Current decoded HMM regime |
| `state_probability` | float | $0.0 \le p \le 1.0$ | Posterior confidence in current state |
| `current_state_id` | integer| `0`, `1`, `2`, `3` | Zero-indexed ordered state ID |
| `current_state_name` | string | `'Normal'`, `'Moderate'`, `'Pre-Surge'`, `'Critical Surge'` | Canonical state name |
| `current_state_posteriors` | dict | 4 floats summing strictly to `1.0000` | Posterior probability simplex distribution |
| `rule_warning_level` | string | `'NORMAL'`, `'SURGE ALERT (PRE-SURGE)'`, `'CRITICAL SURGE'` | Warning tier from occupancy thresholds only |
| `hmm_warning_level` | string | `'NORMAL'`, `'SURGE ALERT (PRE-SURGE)'`, `'CRITICAL SURGE'` | Warning tier from HMM state and next-day surge risk |
| `warning_level` | string | Equal to `rule_warning_level` | Operational warning tier (strictly follows rule) |
| `warning_reason` | string | Plain text string | Explains which specific threshold condition fired |
| `surge_label` | string | Canonical display string | Formatted header banner string |
| `is_physical_surge` | boolean| `true` or `false` | True if latest occupancy $\ge 0.90$ |
| `next_day_surge_probability` | float | $0.0 \le p \le 1.0$ | 1-step ahead transition probability to State 3 |
| `latest_occupancy_rate` | float | $0.0 \le O \le 1.0$ | Physical bed occupancy rate (capped at 1.0) |
| `latest_occupancy_demand`| float | $O \ge 0.0$ | Raw incoming occupancy demand |
| `unmet_demand_beds` | integer| $\ge 0$ | Bed deficit count if demand > 1.0 |
| `recommended_action` | string | Protocol text | Clinical standard operating procedure |

- **Example JSON Request**:
```json
{
  "occupancy_history": [0.74, 0.78, 0.82, 0.86, 0.89, 0.92]
}
```

- **Example JSON Response**:
```json
{
  "status": "success",
  "evidence_level": "experimental",
  "rule_evidence_level": "validated",
  "warning_rule_evidence_level": "validated",
  "hmm_evidence_level": "experimental",
  "state_label": "Pre-Surge",
  "state_probability": 0.5262,
  "current_state_id": 2,
  "current_state_name": "Pre-Surge",
  "current_state_posteriors": {
    "Normal": 0.0000,
    "Moderate": 0.0022,
    "Pre-Surge": 0.5262,
    "Critical Surge": 0.4716
  },
  "rule_warning_level": "CRITICAL SURGE",
  "hmm_warning_level": "CRITICAL SURGE",
  "warning_level": "CRITICAL SURGE",
  "warning_reason": "Current occupancy 92.0% >= 90.0% threshold (CRITICAL SURGE)",
  "surge_label": "SURGE (CRITICAL)",
  "is_physical_surge": true,
  "next_day_surge_probability": 0.4716,
  "latest_occupancy_rate": 0.9200,
  "latest_occupancy_demand": 0.9200,
  "unmet_demand_beds": 0,
  "recommended_action": "CRITICAL SURGE PROTOCOL: Pause elective admissions, activate discharge lounge, open reserve overflow beds."
}
```

---

### 2.4 Endpoint 4: `run_scenario(name: str = "flu_season_rush", capacity: int = 120, multiplier: float = 1.35) -> dict`
Simulates a multi-week operational stress scenario using the queueing simulation engine.

- **Evidence Level**: `"experimental"`
- **Convenience Helper**: `flu_scenario(beds: int = 120, multiplier: float = 1.35) -> dict`
- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'experimental'` | Simulation evidence tier |
| `is_simulated` | boolean| `true` | Simulation flag |
| `average_stay_days` | float | `32.3` | Training stay distribution mean length of stay (days) |
| `beds` | integer| Physical bed count | Configured licensed bed capacity |
| `multiplier` | float | e.g. `1.35` | Admission arrival rate multiplier |
| `baseline_occupancy` | float | $[0.0, 1.0]$ | Baseline steady-state occupancy (e.g. `0.7665`) |
| `surge_occupancy` | float | $[0.0, 1.0]$ | Mean occupancy during surge (e.g. `0.9768`) |
| `occupancy_change_pp` | float | e.g. `21.03` | Percentage points increase over baseline |
| `days_above_90pct` | integer| $[0, 60]$ | Days in surge regime ($\ge 90\%$) |
| `pct_days_above_90` | float | e.g. `76.7` | Percentage of surge days above 90% |
| `peak_occupancy` | float | $[0.0, 1.0]$ | Peak physical bed occupancy (capped at 1.0) |
| `peak_occupancy_demand`| float | e.g. `1.1333` | Peak total patient demand rate (113.3%) |
| `peak_unmet_demand_beds`| integer| e.g. `16` | Peak overflow beds required |
| `unmet_demand_beds` | integer| e.g. `16` | Peak overflow beds required |
| `surge_shortfall_days` | integer| $[0, 60]$ | Total days with bed deficits (e.g. `28`) |

- **Example JSON Response (`run_scenario`)**:
```json
{
  "status": "success",
  "evidence_level": "experimental",
  "is_simulated": true,
  "average_stay_days": 32.3,
  "beds": 120,
  "multiplier": 1.35,
  "baseline_occupancy": 0.7665,
  "surge_occupancy": 0.9768,
  "occupancy_change_pp": 21.03,
  "days_above_90pct": 46,
  "pct_days_above_90": 76.7,
  "peak_occupancy": 1.0,
  "peak_occupancy_demand": 1.1333,
  "peak_unmet_demand_beds": 16,
  "unmet_demand_beds": 16,
  "surge_shortfall_days": 28
}
```

---

### 2.5 Endpoint 5: `predict_ward(patients: list[dict], capacity: int = None) -> dict`
Aggregates individual stay predictions across a hospital ward cohort, leveraging the Central Limit Theorem error-cancellation effect.

- **Evidence Level**: `"validated"`
- **Input Parameters**:
  - `patients` (list[dict]): List of patient attribute dictionaries currently admitted to the ward.
  - `capacity` (integer, optional): Licensed bed capacity of the ward.
- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'validated'` | Scientific rigor tier |
| `patient_count` | integer| $N \ge 0$ | Total patients in ward cohort |
| `total_expected_bed_days`| float| $\ge 0.0$ | Sum of soft Bayesian expected stay days from admission ($E[D_i]$) |
| `mean_stay_days` | float | $\ge 0.0$ | Cohort average length of stay (days) |
| `typical_error_pct` | float | e.g. `8.51` | Expected MAPE interpolated from ward summing evaluation |
| `reliable` | boolean| `true` or `false` | True if cohort $N \ge 20$ (adequate CLT error cancellation) |
| `note` | string | Text note | Clarifies total_expected_bed_days represents full stay from admission |
| `long_stay_alerts_count` | integer| $0 \le n \le N$ | Number of patients triggering Long-Stay Alert ($\theta=0.25$) |
| `configured_capacity` | integer| Optional | Licensed ward bed capacity |
| `patients` | list[dict] | List of patient dicts | Full individual patient prediction objects |

- **Example JSON Request**:
```json
{
  "capacity": 20,
  "patients": [
    {"Department": "radiotherapy", "Severity of Illness": "Extreme", "Age": "51-60", "Visitors with Patient": 6, "Admission_Deposit": 6200.0, "Type of Admission": "Emergency"}
  ]
}
```

- **Example JSON Response (Summary for $N=20$)**:
```json
{
  "status": "success",
  "evidence_level": "validated",
  "patient_count": 20,
  "total_expected_bed_days": 650.0,
  "mean_stay_days": 32.3,
  "typical_error_pct": 8.51,
  "reliable": true,
  "note": "total_expected_bed_days represents the full length of stay from admission for all cohort patients, not remaining stay days.",
  "long_stay_alerts_count": 3,
  "configured_capacity": 20
}
```

---

### 2.6 Endpoint 6: `get_evidence() -> dict`
Returns self-documenting headline validation benchmarks and clinical limitations sourced directly from `reports/metrics.json`.

- **Evidence Level**: `"validated"`
- **Input Parameters**: None
- **Output Response Schema**:

| Response Key | Type | Valid Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `status` | string | `'success'` | API execution status |
| `evidence_level` | string | `'validated'` | Scientific rigor tier |
| `stay_accuracy_vs_baseline`| dict | 11-band & 3-class metrics | Inpatient stay model benchmarks vs majority baselines |
| `long_stay_alert` | dict | Alert engine metrics | Recall (77.51%), precision (60.27%), ROC-AUC (0.8956) at $\theta=0.25$ |
| `ward_error_by_n` | list[dict] | Evaluations at $N \in [1, 20, 50, 100]$ | CLT error-cancellation curve (56.54% down to 3.84% MAPE) |
| `hmm_vs_85_rule` | dict | Comparison metrics | Operational verdict, dwell times (~57d), and 85% rule recommendation |
| `forecast_vs_persistence` | dict | Forecasting comparisons | Daily MAEs (2.03 vs 1.64 pp), 73.3% CI coverage, and persistence verdict |
| `sparcs_result` | dict | External generalization | Majority baseline performance (89.05% vs 89.09% baseline) |
| `limitations` | list[str]| 5 clinical limitation strings | Non-Indian data, simulated occupancy, Kaggle stay bands, HHS series |

- **Example JSON Response**:
```json
{
  "status": "success",
  "evidence_level": "validated",
  "stay_accuracy_vs_baseline": {
    "11_band": {
      "model_name": "Artificial Neural Network (MLP 64x32, Calibrated Isotonic)",
      "accuracy": 0.4249,
      "baseline_accuracy": 0.2748,
      "delta_pp": 15.01,
      "brier_score": 0.7123,
      "macro_roc_auc": 0.8078
    },
    "3_class": {
      "accuracy": 0.6263,
      "baseline_accuracy": 0.4848,
      "delta_pp": 14.15,
      "macro_f1": 0.6228,
      "macro_roc_auc": 0.7958
    }
  },
  "long_stay_alert": {
    "threshold": 0.25,
    "target_class": "Long (>50d)",
    "recall": 0.7751,
    "precision": 0.6027,
    "f1_score": 0.6782,
    "roc_auc": 0.8956
  },
  "ward_error_by_n": [
    {"n": 1, "mape_pct": 56.54, "clt_bound_pct": 56.54, "bias_pct": 0.09, "reliable": false},
    {"n": 20, "mape_pct": 8.51, "clt_bound_pct": 12.64, "bias_pct": 0.53, "reliable": true},
    {"n": 50, "mape_pct": 5.42, "clt_bound_pct": 8.0, "bias_pct": 0.23, "reliable": true},
    {"n": 100, "mape_pct": 3.84, "clt_bound_pct": 5.65, "bias_pct": 0.47, "reliable": true}
  ],
  "hmm_vs_85_rule": {
    "selected_k": 4,
    "verdict": "no better than the 85% rule (different recall/false-alarm trade-off)",
    "mean_dwell_days": 57.0,
    "rule_85_performance": "100% recall with 11-14 days lead time"
  },
  "forecast_vs_persistence": {
    "verdict": "patient-based Monte Carlo: does not work; simple mean-reversion rule on HHS: roughly equal to persistence",
    "day_1_mae_pp": {"blended": 2.03, "persistence": 1.64},
    "day_7_mae_pp": {"blended": 4.62, "persistence": 4.31},
    "empirical_80_ci_coverage": 0.733,
    "coverage_se": 0.081
  },
  "sparcs_result": {
    "ann_accuracy": 0.8905,
    "baseline_accuracy": 0.8909,
    "delta_pp": -0.04,
    "recall_over_10_days": 0.0167
  },
  "limitations": [
    "Non-Indian training data: Trained on US/Kaggle and SPARCS (New York) datasets that differ in demographics, payer structures, and disease profiles.",
    "Simulated occupancy: Bed occupancy is simulated via Little's Law Poisson arrivals and not validated on live hospital census feeds.",
    "32.3-day average stay: Kaggle training distribution has an average stay of 32.3 days (vs 3-5 days in standard Indian acute care wards).",
    "COVID-era HHS timeseries: State-level HHS time series reflects pandemic surge patterns with prolonged ~57-day state dwelling.",
    "Subsampling: Linear SVM and tree depth ablations were trained on 25k-row subsamples for computational tractability."
  ]
}
```

---

## 3. Global Enums, Canonical Label Strings, and Value Mappings

### 3.1 Eleven Length-of-Stay Duration Bands (`STAY_BANDS`)
| Band Index | Band Label String | Day Range (Days) | Mathematical Midpoint (Days) |
| :--- | :--- | :--- | :--- |
| `0` | `'0-10'` | $[0, 10]$ | `5.0` |
| `1` | `'11-20'` | $[11, 20]$ | `15.0` |
| `2` | `'21-30'` | $[21, 30]$ | `25.0` |
| `3` | `'31-40'` | $[31, 40]$ | `35.0` |
| `4` | `'41-50'` | $[41, 50]$ | `45.0` |
| `5` | `'51-60'` | $[51, 60]$ | `55.0` |
| `6` | `'61-70'` | $[61, 70]$ | `65.0` |
| `7` | `'71-80'` | $[71, 80]$ | `75.0` |
| `8` | `'81-90'` | $[81, 90]$ | `85.0` |
| `9` | `'91-100'` | $[91, 100]$ | `95.0` |
| `10` | `'More than 100 Days'` | $[101, \infty)$ | `110.0` |

### 3.2 Three Operational Inpatient Triage Tiers (`CLASS_3_NAMES`)
| Class Index | Tier Label String | Clinical Definition | Stay Band Mapping | Pre-tuned Threshold |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `'Short (0-20d)'` | Short acute stay ($\le 20$ days) | Bands `0-10`, `11-20` | Default $\text{argmax}$ |
| `1` | `'Medium (21-50d)'` | Intermediate stay ($21 - 50$ days) | Bands `21-30`, `31-40`, `41-50` | Default $\text{argmax}$ |
| `2` | `'Long (>50d)'` | Prolonged stay ($> 50$ days) | Bands `51-60` through `More than 100 Days` | Optimized $\theta = 0.25$ |

### 3.3 Four HMM Operational Regimes
| State ID | Canonical State Name | Mean Occupancy | Expected Dwell Time | Operational Description |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `'Normal'` | `66.2%` | `5.5 days` | Routine steady-state operations |
| `1` | `'Moderate'` | `81.31%` | `5.1 days` | Elevated census; active monitoring |
| `2` | `'Pre-Surge'` | `86.65%` | `56.7 days` (~57d) | Buffer exhaustion; pre-surge alert |
| `3` | `'Critical Surge'` | `90.02%` | `58.0 days` (~58d) | Physical capacity saturation ($\ge 90\%$) |

### 3.4 Unified Operational Warning Hierarchy & Standard Operating Protocols
The primary operational triage status (`warning_level`) is strictly determined by the validated occupancy threshold rules (`rule_warning_level`). The Hidden Markov Model provides parallel experimental regime monitoring (`hmm_warning_level`).

| Warning Tier String | Visual Color | Logical Trigger Condition | Mandatory Clinical Protocol String |
| :--- | :--- | :--- | :--- |
| **`"CRITICAL SURGE"`** | **RED** | Current Occupancy $\ge 0.90$ | `"CRITICAL SURGE PROTOCOL: Pause elective admissions, activate discharge lounge, open reserve overflow beds."` |
| **`"SURGE ALERT (PRE-SURGE)"`** | **YELLOW** | Current Occupancy $\in [0.85, 0.90)$ | `"SURGE ALERT PROTOCOL: Notify on-call nursing teams, expedite discharge rounds, review step-down candidates."` |
| **`"NORMAL"`** | **GREEN** | Current Occupancy $< 0.85$ | `"NORMAL OPERATIONS: Standard patient intake and routine discharge rounds."` |

### 3.5 Three Clinical Phenotype Clusters
| Cluster ID | Canonical Label String | Mean Stay | Top Clinical Characteristics |
| :--- | :--- | :--- | :--- |
| `0` | `'Cluster A'` | `33.58 days` | Elderly cohort (mean age 61.2y, deposit $5,463); 57.2% trauma, 66.8% gynecology. |
| `1` | `'Cluster B'` | `31.66 days` | High physiological acuity (33.9% extreme severity, deposit $4,533, age 43.6y). |
| `2` | `'Cluster C'` | `30.68 days` | Younger acute oncology/emergency cohort (age 41.5y, 42.1% radiotherapy, 48.3% emergency). |
