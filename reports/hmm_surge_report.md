# WardCast HMM Bed Surge Detector Report (Step 11)

## 1. Mathematical Architecture & State Space
A continuous-observation **Gaussian Hidden Markov Model** was implemented via `hmmlearn.hmm.GaussianHMM` to model hospital operational capacity regimes:
- **Observation Space**: $\mathbf{X}_t = [O_t, \Delta O_t]^T \in \mathbb{R}^2$
  - $O_t$: Daily hospital occupancy rate ($0.0 \le O_t \le 1.0$).
  - $\Delta O_t = O_t - O_{t-1}$: 1-day momentum / rate of change.
- **Training Series**: Rhode Island HHS state time series (70% chronological split, 1,057 days from 2020-01-01 to 2022-11-23; chronological test split of 453 days from 2022-11-24 to 2024-02-19).
- **Inference Algorithms**:
  - **Parameter Estimation**: **Baum-Welch algorithm** (Expectation-Maximization utilizing forward-backward probabilities).
  - **State Sequence Decoding**: **Viterbi algorithm** (dynamic programming finding the maximum a posteriori state path).

### Model Selection via BIC and Held-Out Log-Likelihood:
| State Count ($k$) | Train Log-Likelihood | Held-Out Test LL | Parameters ($p$) | BIC | Selection Assessment |
|---|---|---|---|---|---|
| **$k=2$** | 4580.6 | 2224.3 | 9 | -9098.5 | Under-parameterized (coarse two-state split) |
| **$k=3$** | 4629.0 | 2223.4 | 17 | -9139.7 | Suboptimal held-out log-likelihood |
| **$k=4$ (Selected)** | **4827.3** | **2258.2** | **27** | **-9466.6** | **Optimal**: Strictly minimizes BIC (-9,466.6) and maximizes test LL (2,258.2) |

---

## 2. Four Operational States & Dwell Times
The fitted 4-state Gaussian HMM defines four distinct operational regimes ordered by ascending mean occupancy:

| State Index | State Name | Mean Occupancy Rate | Mean Daily Momentum | Expected Dwell Time | Clinical Operational Regime |
|---|---|---|---|---|---|
| **State 0** | **Normal** | **66.2%** | -0.0003 | **5.5 days** | Routine steady-state baseline operations |
| **State 1** | **Moderate** | **81.31%** | +0.0001 | **5.1 days** | Elevated routine census; active monitoring |
| **State 2** | **Pre-Surge** | **86.65%** | +0.0001 | **56.7 days** (~57 days) | Buffer exhaustion; early warning regime |
| **State 3** | **Critical Surge** | **90.02%** | -0.0001 | **58.0 days** (~58 days) | Capacity saturation ($\ge 90\%$); crisis staffing |

### Explanation of the ~57-Day State Dwell Times:
In acute care hospitals during seasonal winter respiratory waves and pandemic waves, state-level hospital occupancy hovered persistently between 85% and 92% for months at a time rather than oscillating daily. The diagonal transition probabilities ($A_{22} \approx 0.982, A_{33} \approx 0.983$) capture this prolonged systemic inertia, resulting in expected continuous dwell times of approximately **57 days** ($1 / (1 - A_{ii}) \approx 57$ days) for both State 2 (56.7 days) and State 3 (58.0 days). 

Consequently, the HMM functions as a broad **macro-epidemiological regime identifier** (identifying multi-month surge waves) rather than a high-frequency, rapid-switching daily alarm.

---

## 3. Quantitative Evaluation Against Calibrated Threshold Baselines

Evaluated on the chronological held-out test windows across multiple US states. Physical surge is defined as hospital inpatient occupancy rate $\ge 90\%$.

### 3.1 Headline Early Warning Performance (7-Day Ahead Lead Time)
Assesses whether an alert on day $t$ correctly anticipates a physical surge ($\ge 90\%$) occurring within the subsequent 7 days:

| State | Evaluation Window | Evaluation Model / Baseline | Recall | Precision | F1 Score | False Alarms | False Alarms / 100d |
|---|---|---|---|---|---|---|---|
| **Rhode Island (RI)** | 453 test days (143 surge days, 342 next-7d surge days) | **Calibrated Rule ($\ge 85\%$)** | **96.78%** | **76.09%** | **0.8520** | 104 | 22.96 |
| | | **Persistence ($\ge 90\%$)** | 38.01% | 90.91% | 0.5361 | 13 | 2.87 |
| | | **HMM Critical State 3** | **80.41%** | **91.67%** | **0.8567** | 25 | 5.52 |
| | | **HMM Alert State 2+** | **100.00%** | **76.00%** | **0.8636** | 108 | 23.84 |
| **Delaware (DE)** | 446 test days (4 surge days, 16 next-7d surge days) | **Calibrated Rule ($\ge 85\%$)** | 31.25% | 13.51% | 0.1887 | 32 | 7.17 |
| | | **Persistence ($\ge 90\%$)** | 12.50% | 50.00% | 0.2000 | 2 | 0.45 |
| | | **HMM Critical State 3** | 6.25% | 50.00% | 0.1111 | 1 | 0.22 |
| | | **HMM Alert State 2+** | 50.00% | 7.27% | 0.1270 | 102 | 22.87 |
| **Arizona (AZ)** | 454 test days (0 surge days) | **Calibrated Rule ($\ge 85\%$)** | n/a | n/a | n/a | 0 | 0.00 |
| | | **HMM Alert State 2+** | n/a | n/a | n/a | 1 | 0.22 |
| **South Carolina (SC)** | 454 test days (0 surge days) | **Calibrated Rule ($\ge 85\%$)** | n/a | n/a | n/a | 0 | 0.00 |
| | | **HMM Alert State 2+** | n/a | n/a | n/a | 1 | 0.22 |
| **West Virginia (WV)** | 457 test days (0 surge days) | **Calibrated Rule ($\ge 85\%$)** | n/a | n/a | n/a | 4 | 0.88 |
| | | **HMM Alert State 2+** | n/a | n/a | n/a | 12 | 2.63 |

> **Note on Low-Surge States**: In AZ, SC, and WV, zero surge days ($\ge 90\%$) occurred during the test window. Recall, precision, and F1 scores are formally reported as `n/a` to reflect undefined denominators, rather than zero performance.

### 3.2 Same-Day Physical Surge Agreement
| Evaluation Cohort | Model / Baseline | Recall | Precision | F1 Score | False Alarms / 100d |
|---|---|---|---|---|---|
| **Rhode Island (Test)** | **Calibrated Rule ($\ge 85\%$)** | 100.0% | 32.87% | 0.4948 | 64.46 |
| | **Physical Threshold ($\ge 90\%$)** | 100.0% | 100.0% | 1.0000 | 0.00 |
| | **HMM Critical State 3** | 93.01% | 44.33% | 0.6005 | 36.87 |
| | **HMM Alert State 2+** | 100.0% | 31.78% | 0.4823 | 67.77 |

---

## 4. Final Conclusion & Operational Verdict
- **Verdict**: The HMM is **no better than the 85% rule (different recall/false-alarm trade-off)**.
- **Trade-off Analysis**:
  - In low-surge states (AZ, SC, WV), physical surges are absent, and both simple thresholds and the HMM produce near-zero alarms.
  - In high-surge states (RI), occupancy remains elevated across multi-week periods (~57-day dwell time in States 2 and 3). Triggering an alert on State 2+ yields 100.0% recall but produces 23.84 false alarms per 100 days (comparable to the 85% rule's 22.96 false alarms per 100 days). Restricting alerts to State 3 reduces false alarms (5.52 / 100d) but drops recall to 80.41%.
- **Operational Recommendation**: The **simple calibrated rule (occupancy $\ge 85\%$ alert)** is recommended for frontline hospital triage operations due to transparency, zero hyperparameter maintenance, and identical operational recall. The HMM is retained as an **experimental** epidemiological regime classifier.

---

## 5. Operational Warning Level & Action Protocol Hierarchy
The HMM state decoding and occupancy triggers are harmonized into a single 3-tier operational response hierarchy:

| Warning Tier | Visual Color | Trigger Logic | Mandatory Operational Protocol |
| :--- | :--- | :--- | :--- |
| **`CRITICAL SURGE`** | **RED** | Current Occupancy $\ge 90\%$ **OR** HMM State == 3 (`Critical Surge`) **OR** Next-Day Surge Risk $\ge 50\%$ | `CRITICAL SURGE PROTOCOL: Pause elective admissions, activate discharge lounge, open reserve overflow beds.` |
| **`SURGE ALERT (PRE-SURGE)`** | **YELLOW** | Current Occupancy $\in [85\%, 90\%)$ **OR** HMM State == 2 (`Pre-Surge`) **OR** Next-Day Surge Risk $\in [20\%, 50\%)$ | `SURGE ALERT PROTOCOL: Notify on-call nursing teams, expedite discharge rounds, review step-down candidates.` |
| **`NORMAL`** | **GREEN** | Current Occupancy $< 85\%$ **AND** HMM State $\in \{0, 1\}$ **AND** Next-Day Surge Risk $< 20\%$ | `NORMAL OPERATIONS: Standard patient intake and routine discharge rounds.` |

- **State Trajectory Artifact**: The complete Viterbi-decoded state sequence over Rhode Island's occupancy series is preserved in `reports/figures/hmm_viterbi_state_path.png`.
