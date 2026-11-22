# WardCast 7-Day Rolling Census Forecast Evaluation Report (Step 12)

## 1. Engine Architecture & Mathematical Formulation
The WardCast 7-day rolling forecast executes a patient-tracking discrete-event Monte Carlo simulation (1,000 runs):
1. **Patient-Specific Conditional Survival**:
   For each inpatient $i$ with $s_i$ elapsed days in hospital, future stay probability is evaluated under Bayes' rule:
   $$P(D_i > s_i + h \mid D_i > s_i) = \frac{S_i(s_i + h)}{S_i(s_i)}$$
   where $S_i(d) = \sum_{b=0}^{10} p_{i, b} \cdot P(\text{Stay} > d \mid \text{Band}_b)$, using probability vectors $p_i$ from `models/stay_model.joblib`.
2. **Poisson Inpatient Arrivals**:
   Prospective daily admissions follow $\text{Poisson}(\lambda_w)$ calibrated on rolling 4-week weekday history.
3. **Capacity & Surge Tiering**:
   Produces daily median census, 80% prediction intervals ($[q_{10}, q_{90}]$), capacity overflow shortfall, and warning tiers.

---

## 2. Walk-Forward Backtest Performance vs. Baselines (Horizons 1 to 7)
Evaluated across **60 walk-forward forecast origins** (120 beds) spanning all seasons with zero temporal shuffling.

| Horizon (Days) | Pure MC Model MAE (pp) | Blended Model MAE (pp) | Persistence MAE (pp) | 7D MA MAE (pp) | Seasonal-Naive MAE (pp) | Margin vs Persist (pp) | 95% CI Margin vs Persist (pp) | Pure Model Verdict | Blended Model Verdict | Test Window 80% CI Coverage (%) | Coverage Standard Error (%) | Surge Events Observed | Surge Metric Label |
|:---|---:|---:|---:|---:|---:|---:|:---|:---|:---|---:|---:|---:|:---|
| Day +1 | **3.19** | **2.03** | **1.64** | **2.49** | **4.10** | +0.39 | [-0.02, +0.81] | does not beat baselines | does not beat baselines | **70.0%** | 8.4% | 4 | too few events to judge (N=4) |
| Day +2 | **3.48** | **2.70** | **1.85** | **2.82** | **4.03** | +0.85 | [+0.03, +1.67] | does not beat baselines | does not beat baselines | **73.3%** | 8.1% | 3 | too few events to judge (N=3) |
| Day +3 | **3.90** | **3.38** | **2.44** | **2.96** | **3.74** | +0.93 | [-0.11, +1.97] | does not beat baselines | does not beat baselines | **73.3%** | 8.1% | 4 | too few events to judge (N=4) |
| Day +4 | **4.00** | **3.76** | **3.04** | **3.43** | **3.50** | +0.72 | [-0.51, +1.95] | does not beat baselines | does not beat baselines | **70.0%** | 8.4% | 1 | too few events to judge (N=1) |
| Day +5 | **3.91** | **3.73** | **3.28** | **3.64** | **3.67** | +0.45 | [-0.74, +1.64] | does not beat baselines | does not beat baselines | **70.0%** | 8.4% | 4 | too few events to judge (N=4) |
| Day +6 | **4.34** | **4.23** | **3.75** | **4.11** | **3.97** | +0.48 | [-0.73, +1.69] | does not beat baselines | does not beat baselines | **80.0%** | 7.3% | 4 | too few events to judge (N=4) |
| Day +7 | **4.68** | **4.62** | **4.31** | **4.62** | **4.31** | +0.32 | [-1.06, +1.70] | does not beat baselines | does not beat baselines | **76.7%** | 7.7% | 2 | too few events to judge (N=2) |

---

## 3. Honest Diagnosis: Where the Model Wins and Where Baselines Compete
1. **Short Horizons (Day +1 to Day +3 — Point Error vs Autocorrelation)**:
   - At **Day +1**, Persistence achieves lower point MAE (**1.64 pp**) due to strong 24-hour census autocorrelation. Pure Monte Carlo yields **3.19 pp** MAE, while Blended Mean-Reversion variant narrows this gap to **2.03 pp**.
   - Pure Monte Carlo does not beat 24-hour persistence at any horizon for point MAE.
2. **Medium to Long Horizons (Day +4 to Day +7 — Baseline Convergence)**:
   - At **Day +7**, Pure Model MAE is **4.68 pp** (Blended: **4.62 pp**) vs. 7-Day MA of **4.62 pp**, Seasonal-Naive of **4.31 pp**, and Persistence of **4.31 pp**.
   - Margin of difference between the blended model and persistence is not statistically significant (95% CI [-1.06, +1.70]).
3. **Uncertainty Calibration & Coverage**:
   - The 80% prediction interval achieves **73.3%** average empirical coverage on the held-out test window (standard error: **8.1%**), moderately undercovering the nominal 80% boundary.
4. **Surge Metric Sample Size Limitation**:
   - In the 60-origin walk-forward backtest, physical surge events ($\ge 90\%$) occurred on only **1 to 4 occasions per horizon**, meaning surge classification metrics have wide confidence intervals and are formally labeled `too few events to judge (N < 20)`.
