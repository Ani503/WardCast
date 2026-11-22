# WardCast: Machine Learning Final Summary & Operating Guide

**Course**: PCCAIML 502 (Introduction to Machine Learning)  
**Deliverable**: Machine-Learning Core & Simulation Engine (No UI Scope)  
**Verification**: All figures verified directly against `reports/metrics.json` and benchmark tables.

---

## 1. Component Performance & Verdict Scorecard

| Component | What It Does | Key Measured Number | Baseline Compared With | Verdict | Justification from Measured Numbers |
|---|---|---|---|---|---|
| **Length-of-Stay Prediction** | Predicts inpatient stay duration across 11 ordinal bands and 3 operational tiers | **42.49%** 11-band accuracy (**62.63%** 3-class accuracy) | Majority Baseline: 27.48% (11-band) / 48.48% (3-class) | **Works with caveats** | Outperforms majority baseline by **+14.15 pp** on 3-class target, but single-patient accuracy (42.49%) is modest; reliable primarily when aggregated. |
| **Long-Stay Alert Engine** | Early warning flag for extended stays (>50 days) at tuned threshold $\theta = 0.25$ | **77.51%** recall at 60.27% precision | Default threshold $\theta = 0.50$ (61.22% recall) | **Works** | Raises sensitivity by **+16.29 pp**, capturing >3/4 extended-stay patients with acceptable 60.27% precision. |
| **Probability Calibration** | Calibrates out-of-fold probabilities via isotonic regression (`CalibratedClassifierCV`) | **0.7123** 11-band Brier score (**1.5224** log-loss) | Majority Brier: **1.4505**; Class Prior: **0.8135** (log-loss 1.8982) | **Works** | Better than class-prior (Brier 0.71 vs 0.81, log-loss 1.52 vs 1.90); isotonic vs plain ANN is a negligible difference. |
| **SPARCS stay-length mixture (89% of stays <=10 days)** | Models right-censored tail for stays >100 days using interval-censored EM mixture | **k=6** components, BIC = **9,455,310** | Uncensored variance-floor models (spikes) | **Works** | Strictly minimizes BIC (9,455,310.4) and eliminates single-integer spike collapse ($\sigma \in [0.201, 0.920]$, max 0.92); simulator draws Kaggle >100d tail via truncated lognormal ($\mu=4.75, \sigma=0.15$) in `src/simulator.py` lines 103–108. |
| **Inpatient Clustering** | Identifies operational subgroups (Cluster A, Cluster B, Cluster C) via K-Means / BIRCH | Silhouette = **0.0667**, Davies-Bouldin = **3.1135** | Random partitioning (Silhouette ~0.0) | **Works with caveats** | Low silhouette score (<0.10) indicates **weakly separated groups** with high clinical overlap; stay differences are under 3 days (spread 2.90 days). |
| **Association Rules** | Mines clinical and administrative correlates of extended stays via Apriori / FP-Growth | Top rule lift = **3.59x** (70.15% confidence) | Unconditional base rate (19.56% long stays) | **Works** | Apriori and FP-Growth yield identical 2,752 frequent itemsets; rules are strictly statistical non-causal associations. |
| **External Generalization** | Evaluates hospital-held-out transfer on NY SPARCS 2014 discharges | **89.05%** accuracy on SPARCS | Majority Baseline: **89.09%** | **Does not work** | Performance is at baseline level; band accuracy is dominated by the 89% short-stay class. Direct transfer collapsed to 4.09% acc. |
| **Bed Queueing Simulation** | Simulates 3-year daily hospital census under Little's Law with seasonal arrivals | Simulated Mean Occ = **76.65%**, Volatility = **1.91%** | HHS Target: 74.50% Occ (2.15 pp gap), 2.28% Vol (0.37 pp gap) | **Works with caveats** | Re-calibrated winter arrival multiplier (1.12x) matches HHS seasonal dynamics; baseline peak occ is 0.95 with 0 shortfall days vs 100.0% peak bed occupancy in flu scenario with 16 overflow beds of unmet demand (113.3% total patient demand = 120 licensed beds + 16 overflow demand), rising from 76.65% to 97.68% (+21.03 pp). |
| **HMM Surge Detection** | Unsupervised 4-state Gaussian HMM trained on Rhode Island HHS series | RI Early Warning: Recall = **80.4%**, Precision = **91.7%**; Same-Day: Recall = **93.0%**, Precision = **44.3%** | Calibrated Rule ($\ge 85\%$): Recall = 96.8% (EW), Precision = 76.1%; Persistence ($\ge 90\%$): Recall = 38.0% | **Does not work** | The HMM is no better than the 85% rule (different recall/false-alarm trade-off); we recommend the simple 85% rule for operational triage. Only RI and DE had events. |
| **7-Day Patient Monte Carlo Forecast** | Rolling patient-based Monte Carlo census forecast with calibrated prediction intervals | Pure MC Day 1 MAE = **3.19 pp** (Blended = **2.03 pp**) | Persistence Day 1 MAE = **1.64 pp**; 7D MA = **2.49 pp** | **Does not work** | Patient-based Monte Carlo: does not work; pure model does not beat persistence on point MAE due to 24h autocorrelation. Blended variant closes gap to 2.03 pp. Held-out test coverage is 73.3% (SE 8.1%). |
| **7-Day Mean-Reversion Forecast on HHS** | Simple mean-reversion autoregressive rule ($\rho = 0.95^h$) on aggregate HHS census | Day 1 MAE = **1.44 pp**, Day 7 MAE = **1.68 pp** | Persistence: Day 1 **1.47 pp**, Day 7 **1.69 pp**; Seasonal-Naive: Day 6 **1.69 pp**, Day 7 **1.69 pp** | **Roughly equal to persistence** | Simple mean-reversion rule on HHS: roughly equal to persistence; beats persistence on Day 1 (-0.029 pp), loses to seasonal-naive at Day 6, ties persistence and seasonal-naive at Day 7 (CI [-0.07, +0.04]). |
| **Ward Error Cancellation** | Aggregates individual patient stay predictions across hospital wards | Single Patient: **56.54%** MAPE $\to$ 100-Bed Ward: **3.84%** MAPE | Overall Mean: 87.69% $\to$ 5.27%; Dept Mean: 87.65% $\to$ 5.26%; Benchmark $e_1 / \sqrt{N}$: 5.65% | **Works** | Aggregation cancels individual errors dramatically (+27.1% rel improvement over mean baseline at N=100); 8.75% in Gynecology vs 8.51% in random at N=20. |

---

## 2. What You Can Safely Show in the Demo

1. **Patient Stay Risk Profiling**:
   - Show `predict_stay(patient)` outputting 11-band probabilities, expected stay days (e.g. 32.4 days), and the top 3 driving factors (e.g. Extreme Severity, Trauma Admission, Visitors $\ge 5$).
   - Safely highlight the **Long-Stay Alert Flag** ($\theta = 0.25$), explaining that catching extended stays (>50 days) with 77.51% sensitivity is clinically actionable for bed turnover planning.
2. **Ward-Level Error Cancellation**:
   - Show that while predicting individual stay length has a 56.54% error, summing 50 to 100 patient bed-days in a ward reduces error to **3.84% – 5.42%** with near-zero population bias (+0.47%).
   - Explain that soft Bayesian expected days ($P \cdot \text{midpoints}$) completely eliminates the truncation bias of discrete mode selection.
3. **Exploratory 7-Day Inpatient Fan Chart**:
   - Show the 7-day rolling forecast fan chart displaying the median expected census with 80% prediction intervals (empirically calibrated on early window, achieving 73.3% test window coverage with SE 7.3% - 8.4%).
   - Clarify that neither the pure MC nor the blended model beats persistence on Day 1 point MAE (1.64 pp vs 2.03 pp vs 3.19 pp).
4. **Winter Flu Surge Stress Test**:
   - Demonstrate `flu_scenario(beds=120, multiplier=1.35)`, showing how a 35% surge in respiratory arrivals increases mean occupancy by **+21.03 percentage points** (from 76.65% to 97.68%) and pushes 76.7% of days above the 90% critical threshold with 28 shortfall days, 100.0% peak bed occupancy, and 16 beds of unmet demand (113.3% total patient demand = 120 licensed beds + 16 overflow demand beds).

---

## 3. What You Must NOT Claim

1. **DO NOT claim the ML stay model can replace clinical judgment for individual patient discharge dates.**
   Single-patient accuracy is 42.49% (11-band). It is a statistical capacity planning tool, not a clinical discharge guarantee.
2. **DO NOT claim the HMM is superior to simple threshold rules for surge detection.**
   Real HHS evaluations show that the HMM is no better than the 85% rule (different recall/false-alarm trade-off), with high false alarms in continuous surge states. Recommend the simple calibrated rule (occupancy $\ge 85\%$).
3. **DO NOT claim the 7-day Monte Carlo forecaster beats persistence on next-day point census.**
   Hospital census has extreme 24-hour autocorrelation; today's census is always the strongest point predictor for tomorrow (1.64 pp vs 3.19 pp MAE). The model's value lies in multi-day turnaround forecasting and interval trajectories.
4. **DO NOT claim models "generalize across health systems without re-training".**
   Direct transfer to NY SPARCS collapsed to 4.09% accuracy because 89.09% of SPARCS stays are $\le 10$ days versus only 7.4% in Kaggle. Models must be re-calibrated to each facility's baseline case mix.
5. **DO NOT claim patient clusters are distinct, isolated cohorts.**
   Silhouette scores are 0.0667 – 0.0783 and Davies-Bouldin indices are >3.0; they are weakly separated groups with high internal overlap and stay differences under 3 days.
6. **DO NOT claim association rules or decision tree splits are causal.**
   Rules like `{Visitors >= 5} => {Stay > 50d}` are statistical correlations, not direct causes.
7. **DO NOT claim the 7-day fan chart intervals are fully validated.**
   Empirical coverage on the held-out test window is 73.3% (with SE 7.3% - 8.4%), moderately undercovering the nominal 80% boundary.

---

## 4. Known Methodological Limitations

1. **Non-Indian Training Data & Regional Transferability**: Models were developed using the US-based Kaggle Healthcare Analytics II dataset and validated against NY SPARCS 2014 discharges. These data sources represent American tertiary care workflows, insurance deposit billing structures ($USD), and administrative practices. They are not calibrated to the Indian public or private healthcare ecosystem (e.g., AIIMS, state civil hospitals, or private hospital networks), which operate under different admission volumes, disease epidemiology (e.g. seasonal vector-borne surges), bed-turnover pressures, and patient family lodging dynamics. Local clinical recalibration is essential prior to international deployment.
2. **Simulated Occupancy & Queueing Sizing**: Daily hospital bed occupancy trajectories and surge frequencies are modeled via a discrete-event queueing simulation under Little's Law with Poisson arrivals and seasonal multipliers. The simulator is calibrated to state-level aggregate HHS benchmarks and Little's Law steady-state dynamics; it is not driven by live hospital EHR telemetry or real-time clinical bed-tracking feeds.
3. **Kaggle 32.3-Day Average Stay Disparity**: The Kaggle training dataset duration bands produce an artificial average stay of approximately 32.3 days (32.26 days), heavily skewed toward prolonged tertiary care (only 7.4% $\le 10$ days). In real-world acute hospital care (such as NY SPARCS 2014), the true epidemiological distribution is heavily front-loaded, with 89.09% of admissions $\le 10$ days and a median stay of ~3 days. Models trained on Kaggle reflect specialized tertiary cohorts and cannot be deployed to general community hospitals without facility-specific recalibration.
4. **COVID-Era HHS Timeseries Data**: The empirical hospital time series used for HMM training and backtesting spans from 2020 to 2024 (2020-01-01 to 2024-04-20, covering 80,983 valid records). This historical window was heavily impacted by COVID-19 pandemic admission waves, mandated elective surgery suspensions, ICU conversions, and unprecedented respiratory bed stress, which may not represent peacetime seasonal hospital equilibrium.
5. **Linear SVM 25,000-Row Subsampling**: Due to $O(N^2)$ quadratic memory and solver scaling constraints, the Linear SVM benchmark was evaluated on a 25,000-row patient subsample (of 254,750 training set fold rows), achieving 37.22% benchmark accuracy.
6. **Right-Censored 110-Day Top Band & Simulator Tail ASSUMPTION**: The Kaggle top band (`More than 100 Days`) required an assumed midpoint of 110.0 days for regression and ward-summing tests (creating right-truncation bias). In contrast, the queueing simulator samples the >100-day tail via a truncated log-normal distribution ($\mu=4.75, \sigma=0.15$ in $[101, 150]$ days) in `src/simulator.py` lines 103–108; this is an operational **ASSUMPTION** (continuous decay), not an empirical EM result from SPARCS.
7. **Absence of Dates in SPARCS**: SPARCS 2014 contains annual discharge totals with no calendar dates or timestamps, precluding real-world calendar time series testing on that asset.
8. **Surge Event Sample Sizes**: In the 60-origin walk-forward backtest, physical surge events ($\ge 90\%$) occurred on only 1 to 4 occasions per horizon, meaning surge classification metrics have wide confidence intervals and are labeled "too few events to judge".
