# WardCast: Machine Learning Core

**WardCast** is an inpatient hospital stay and bed-surge forecaster developed for **PCCAIML 502 (Introduction to Machine Learning)**. It provides end-to-end predictive intelligence for hospital operations: predicting patient-level length of stay (LOS), discovering clinical phenotypes via unsupervised clustering, estimating duration distributions via interval-censored Expectation-Maximization (EM), detecting operational bed surges with Hidden Markov Models, and delivering 7-day rolling probabilistic bed census forecasts via Monte Carlo simulation.

> **Scope Note:** This repository contains the **machine learning core, simulation engine, validation benchmarks, and programmatic API only**. In accordance with the project brief, **no UI, Streamlit app, web page, or dashboard is included**. All predictive capabilities are exposed via clean programmatic entry points in `src/predict.py` for downstream UI integration.

---

## 1. Project Folder Layout

```text
WardCast/
├── Data/
│   ├── train_data.csv                              # Kaggle Healthcare Analytics II (318,438 rows, 92,017 patients)
│   ├── test_data.csv                               # Unlabelled test set (137,057 rows; strictly excluded from scoring)
│   ├── train_data_dictionary.csv                   # Data dictionary for Kaggle columns
│   ├── Hospital_Inpatient_Discharges_...1121.csv   # NY SPARCS 2014 de-identified discharges (2,367,550 rows)
│   ├── COVID-19_Reported_Patient_...eries.csv      # HHS state-level daily bed capacity time series (80,983 valid rows)
│   ├── COVID-19_Reported_Patient_...RAW.csv        # HHS facility-level capacity audit (964,632 weekly reports)
│   └── processed/
│       ├── prep_metadata.json                      # Preprocessing column encoders, scalers, and imputed modes
│       ├── sim_params.json                         # HHS calibrated simulation parameters (Little's law, seasonality)
│       ├── sim_daily_census.csv                    # 3-year simulated hospital daily occupancy trajectory (1,095 days)
│       └── sim_patients.csv                        # 3-year simulated individual patient trajectories (3,069 patients)
├── src/
│   ├── __init__.py
│   ├── audit.py                                    # Phase 0: Full dataset audit and patient-group integrity check
│   ├── prep.py                                     # Phase 1: Leak-free feature pipeline, GroupKFold splits, target encoding
│   ├── stay_models.py                              # Phase 2: 13 classification & regression stay models benchmarked
│   ├── em_clusters_rules.py                        # Phase 3: Unsupervised KMeans/Agglo/BIRCH & Apriori/FP-Growth
│   ├── fit_sparcs_em_censored.py                   # Step B: Interval-censored EM Gaussian Mixture (k=4, variance floor)
│   ├── sparcs_validation.py                        # Step C: SPARCS hospital-held-out GroupKFold validation
│   ├── hhs_calibration.py                          # Phase 5: HHS COVID-19 bed capacity calibration & Little's Law tuning
│   ├── simulator.py                                # Step F: Calibrated discrete-event hospital queueing simulator (120 beds)
│   ├── hmm_surge.py                                # Step D: 4-state Ordered Gaussian HMM surge detector
│   ├── forecast.py                                 # Step E: 7-day rolling Monte Carlo census forecaster with 80% CI
│   ├── test_ward_summing.py                        # Step G: Ward-level error cancellation and same-department tests
│   ├── generate_all_figures.py                     # Generates all publication-quality comparison figures
│   ├── build_report.py                             # Programmatically compiles master markdown report from metrics.json
│   ├── evaluate_all.py                             # Model manifest verification and accuracy audit
│   └── predict.py                                  # Clean programmatic prediction interface (4 core operational APIs)
├── tests/
│   └── test_predict.py                             # 13 automated unit tests (API contracts, simplex sums, report consistency)
├── models/
│   ├── stay_model.joblib                           # Top calibrated stay classifier (MLPClassifier Pipeline)
│   ├── clusterer.joblib                            # Unsupervised clinical phenotyper (KMeans Pipeline, k=3)
│   ├── hmm_surge.joblib                            # 4-state Ordered Gaussian HMM surge detection model
│   └── MODEL_MANIFEST.md                           # Formal model cards, schemas, input/output contracts, and provenance
├── reports/
│   ├── FINAL_SUMMARY.md                            # Executive one-page scorecard (Verdicts, Safe Demos, Limitations)
│   ├── model_accuracy_report.md                    # Master technical accuracy report (100% generated from metrics.json)
│   ├── metrics.json                                # Central machine-readable verified metrics repository
│   ├── decision_log.md                             # Step-by-step decision log maintained before/after every phase
│   ├── progress.md                                 # Sequential milestone tracking log
│   ├── stay_model_comparison.csv                   # Out-of-fold metrics across all 13 stay models
│   ├── forecast_evaluation.csv                     # Walk-forward backtesting metrics on simulated facility
│   ├── forecast_real_hhs_evaluation.csv            # Walk-forward backtesting metrics on real HHS time series
│   ├── ward_summing_evaluation.csv                 # Ward aggregation and intra-department error cancellation data
│   ├── cluster_profiles.md                         # Clinical profiles of weakly separated patient clusters
│   └── figures/
│       ├── stay_model_comparison.png               # Benchmark bar chart of all 13 stay models
│       ├── stay_model_calibration.png              # Multi-class reliability curves (ANN vs Bagging)
│       ├── ward_summing_error_cancellation.png     # Empirical error decay vs e1/sqrt(N) benchmark
│       ├── simulator_vs_hhs_occupancy.png          # Simulated occupancy distribution vs HHS empirical target
│       ├── hmm_surge_trajectory.png                # HHS occupancy time series with decoded HMM regimes
│       ├── forecast_mae_by_horizon.png             # Point MAE comparison vs Persistence and 7D MA
│       ├── forecast_fan_chart.png                  # 7-day rolling forecast fan chart with 80% prediction interval
│       └── sparcs_em_stay_distribution.png         # Interval-censored EM Gaussian Mixture density fit
├── run_demo.py                                     # Standalone console demonstration script (all 4 API functions)
├── requirements.txt                                # Frozen Python package dependencies
└── README.md                                       # Master project documentation
```

---

## 2. Installation & Environment Setup

The project runs on **Python 3.11** using only syllabus-permitted libraries (`scikit-learn`, `hmmlearn`, `mlxtend`, `pandas`, `numpy`, `matplotlib`, `joblib`).

```bash
# 1. Clone repository and navigate into project directory
cd WardCast

# 2. Create and activate a clean Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Ordered Reproduction Commands

Every analytical phase can be executed independently from raw data in `Data/` using standard module execution commands:

| Step | Module Command | Approx Runtime | Deliverables Produced |
|---|---|---|---|
| **Step 0** | `python -m src.audit` | ~15 sec | `reports/00_data_audit.md` (raw counts, patient repetition audit) |
| **Step 1** | `python -m src.prep` | ~10 sec | `Data/processed/prep_metadata.json`, `reports/feature_dictionary.md` |
| **Step 2** | `python -m src.stay_models` | ~90 sec | `models/stay_model.joblib`, `reports/stay_model_comparison.csv` |
| **Step 3** | `python -m src.em_clusters_rules` | ~25 sec | `models/clusterer.joblib`, `reports/cluster_profiles.md` |
| **Step 4** | `python -m src.fit_sparcs_em_censored` | ~15 sec | Interval-censored EM fit, `reports/figures/sparcs_em_stay_distribution.png` |
| **Step 5** | `python -m src.sparcs_validation` | ~20 sec | SPARCS `GroupKFold(Facility ID)` validation, `reports/sparcs_validation.md` |
| **Step 6** | `python -m src.hhs_calibration` | ~10 sec | `Data/processed/sim_params.json`, `reports/hhs_calibration.md` |
| **Step 7** | `python -m src.simulator` | ~12 sec | 3-year queueing simulation, `Data/processed/sim_daily_census.csv` |
| **Step 8** | `python -m src.hmm_surge` | ~8 sec | 4-state Ordered Gaussian HMM, `models/hmm_surge.joblib` |
| **Step 9** | `python -m src.forecast` | ~25 sec | 7-day rolling forecaster backtest, `reports/forecast_evaluation.csv` |
| **Step 10** | `python -m src.test_ward_summing` | ~12 sec | Ward summing & intra-department tests, `reports/ward_summing_evaluation.csv` |
| **Step 11** | `python -m src.generate_all_figures` | ~10 sec | Generates all 8 standardized figures in `reports/figures/` |
| **Step 12** | `python -m src.build_report` | ~2 sec | Programmatically compiles `reports/model_accuracy_report.md` |

---

## 4. Test Suite & Console Demonstration

### Automated Unit Test Suite
The test suite validates programmatic API contracts, probability simplex normalization, 3-class target mappings, and verifies consistency between `reports/metrics.json` and `reports/model_accuracy_report.md`.

```bash
python -m unittest tests/test_predict.py
```
*Expected Result:* 13 tests ran in ~1.6 seconds, 100% passing (`OK`).

### Turnkey Console Demonstration
A standalone CLI demonstration script executes all four operational inference functions with human-readable formatting, percentage displays, ASCII bar visualizers, and zero raw dictionary dumps:

```bash
python run_demo.py
```

The script demonstrates:
1. **`predict_stay(sample_patient)`**: 11-band calibrated probability vector, primary predicted band, top clinical factors, and high-acuity long-stay alert at $\theta=0.25$.
2. **`forecast_7day(current_occupancy=0.78, beds=120)`**: 7-day rolling census trajectory, calibrated 80% prediction intervals, and bed surge risk.
3. **`surge_state(recent_occupancy_history)`**: HMM operational regime decoding (`Normal`, `Busy`, `Surge`), posterior probabilities, next-day surge risk, and protocol recommendations.
4. **`flu_scenario(beds=120, multiplier=1.35)`**: Winter epidemic stress test evaluating surge occupancy (+17.88 pp) and days $\ge 90\%$ capacity.

---

## 5. Master Findings & Component Readiness Scorecard

All metrics are directly measured from code execution and stored in `reports/metrics.json`:

| Component | Primary Syllabus Method | Key Empirical Metric | Reference Baseline | Verdict |
|---|---|---|---|---|
| **Length-of-Stay Prediction** | ANN (`MLPClassifier`, 64x32) | **42.49%** 11-band acc (**62.63%** 3-class acc) | Majority Baseline: 27.48% (11-band) / **48.48%** (3-class) | **Works with caveats** (modest single-patient accuracy; +14.15 pp gain over baseline) |
| **Long-Stay Alert Engine** | Threshold Tuning ($\theta = 0.25$ on ANN) | **77.51%** Long-stay recall | Default $\theta = 0.50$: 61.25% recall | **Works** (+16.26 pp sensitivity gain; catches >3/4 extended-stay patients) |
| **Probability Calibration** | CalibratedClassifierCV (Isotonic) | **0.7123** 11-band Brier score | Majority Brier: **1.4505**; Uniform: **0.9091** | **Works** (best-calibrated engine among tested alternatives) |
| **Stay Distribution Tail** | Interval-Censored EM Gaussian Mixture | **k=4** components, BIC = **9,456,329.9** | Uncensored/Spike floor models | **Works** (eliminates single-integer spike collapse; bounded log-likelihood) |
| **Inpatient Clustering** | K-Means & BIRCH (k=3) | Silhouette = **0.0667**, Davies-Bouldin = **3.1135** | Random Partitioning | **Works with caveats** (weakly separated groups, high clinical overlap) |
| **Frequent Itemset Rules** | Apriori & FP-Growth | Top Rule: **3.59x** lift, **70.15%** confidence | Unconditional Base Rate (19.56%) | **Works** (exact mathematical equivalence between Apriori and FP-Growth; statistical non-causal rules) |
| **External Generalization** | Hospital-Held-Out GroupKFold (SPARCS) | **89.05%** acc | Majority Baseline: **89.09%** | **Does not work** (performance is at baseline level; band accuracy is dominated by the 89% short-stay class) |
| **Bed Queueing Simulation** | Discrete-Event Queueing (Little's Law) | Simulated Mean Occ = **76.65%**, Volatility = **1.91%** | HHS Target: 74.50% Occ (2.15 pp gap), 2.28% Vol (0.37 pp gap) | **Works with caveats** (calibrated to state-level aggregates, not validated on hospital records) |
| **HMM Surge Detection** | Ordered Gaussian HMM (k=4) | RI Test: Rec = **93.0%**, Prec = **44.3%** | Calibrated Rule (>=85%): Rec = 100%, Prec = 32.9% | **Does not work** (does not consistently beat simple threshold rules; 36.9 false alarms/100d; recommended as experimental extra) |
| **7-Day Census Forecast** | Monte Carlo + Poisson Arrivals | Pure MC Day 1 MAE = **3.19 pp** | Persistence Baseline Day 1 MAE = **1.64 pp** | **Does not work** (pure model does not beat baselines; blended variant closes error to 2.03 pp) |
| **Ward Error Cancellation** | Central Limit Theorem Aggregation | Single-Patient: **56.54%** MAPE $\to$ N=100 Ward: **3.84%** MAPE | Single-Patient Benchmark ($e_1 / \sqrt{N}$): 5.65% | **Works** (aggregation cancels individual errors; intra-department wards achieve 5.3% - 5.7% MAPE at N=50) |

---

## 6. Syllabus Compliance Verification

All final models strictly adhere to the **PCCAIML 502 syllabus constraints**:
* **Regression:** Multi-variable Linear Regression, LASSO (`sklearn.linear_model.Lasso`).
* **Classification:** K-Nearest Neighbors (`KNeighborsClassifier`), Categorical Naive Bayes (`CategoricalNB`), Linear Support Vector Machine (`LinearSVC` with Platt scaling), Decision Tree (`DecisionTreeClassifier`).
* **Artificial Neural Networks:** Multi-Layer Perceptron (`MLPClassifier`, `MLPRegressor` with backpropagation).
* **Ensembles:** Bagging (`BaggingClassifier`), Random Forest (`RandomForestClassifier`), AdaBoost (`AdaBoostClassifier`), Gradient Boosting (`GradientBoostingClassifier`).
* **Expectation-Maximization (EM):** Gaussian Mixture Models (`sklearn.mixture.GaussianMixture`).
* **Hidden Markov Models:** Continuous Gaussian HMM (`hmmlearn.hmm.GaussianHMM`) with Forward-Backward posterior decoding and Viterbi state tracking.
* **Association Rules:** Apriori and FP-Growth (`mlxtend.frequent_patterns`).
* **Clustering:** K-Means (`KMeans`), Agglomerative Hierarchical (`AgglomerativeClustering`), BIRCH (`Birch`).
* **Strict Exclusions Upheld:** No XGBoost, LightGBM, CatBoost, LSTM/deep recurrent networks, SARIMA, Prophet, or SHAP were utilized in any final model.
# WardCast
