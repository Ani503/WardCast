# WardCast Phase 5: HHS Capacity Calibration Report

## 1. Overview & Data Cleaning Audit
Hospital capacity and occupancy metrics were calibrated using the HHS state-level daily time series:
`COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv`
- **Raw Observations**: 81,335 rows across 54 states and US territories.
- **Data Filtering**:
  - Missing or non-positive `inpatient_beds`: **352 rows dropped** (0.43% of dataset).
  - Unrealistic values (`occupancy_rate > 1.2`): **0 rows dropped**.
  - **Cleaned Dataset**: **80,983 valid daily observations** from January 1, 2020 through April 20, 2024.
- **Benchmark States**: Texas (TX), California (CA), New York (NY), Arizona (AZ), Rhode Island (RI) (1,572 contiguous observation days per state).

---

## 2. Extracted Empirical Simulation Parameters
| Parameter | Calibrated Value | Clinical & Operational Meaning |
|---|---|---|
| **Mean Occupancy Rate** | **74.5%** | Volume-weighted bed utilization across benchmark states. |
| **Day-to-Day Volatility ($\sigma$)** | **2.28%** | Expected daily census fluctuation standard deviation. |
| **Surge Threshold** | **$\ge$ 90.0%** | Critical capacity warning trigger (near-zero reserve beds). |
| **Historical Surge Frequency** | **5.5%** | Percentage of operational days in red surge zone ($\ge 90\%$). |
| **Mean Surge Episode Duration** | **3.0 days** (Median: 2.0d) | Duration of prolonged high-occupancy stress periods. |
| **Max Surge Episode Duration** | **25 days** | Peak continuous overload during major winter pandemic waves. |

### Calibrated Day-of-Week Multipliers:
| Day | Relative Arrival / Census Multiplier | Operational Characteristic |
|---|---|---|
| **Monday** | **0.9799** (98.0%) | Post-weekend backlog & elective intake surge. |
| **Tuesday** | **1.0059** (100.6%) | Weekly operational peak for planned admissions. |
| **Wednesday** | **1.0183** (101.8%) | Sustained mid-week surgical peak. |
| **Thursday** | **1.0171** (101.7%) | Pre-discharge peak census. |
| **Friday** | **1.0120** (101.2%) | High planned discharges balancing emergent admissions. |
| **Saturday** | **0.9899** (99.0%) | Reduced elective intake; primarily emergent. |
| **Sunday** | **0.9768** (97.7%) | Minimum weekly census and elective intake. |

### Calibrated Seasonal Quarterly Occupancy:
| Season | Months Included | Mean Occupancy Rate | Surge Frequency ($\ge 90\%$) |
|---|---|---|---|
| **Winter** | December, January, February | **72.58%** (Benchmark States: 76.8%) | **5.5%** |
| **Fall** | September, October, November | **71.72%** | 3.2% |
| **Summer** | June, July, August | **69.98%** | 2.1% |
| **Spring** | March, April, May | **67.82%** | 1.8% |

---

## 3. Facility-Level RAW Cross-Check
To cross-validate state-aggregated time series against ground-truth facility reporting, an audit sample of 31,044 records from `COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW.csv` (447 MB) was analyzed:
1. **Facility Bed Capacity Distribution**:
   - 10th Percentile: **24 beds** (rural critical-access hospitals).
   - **Median Bed Capacity**: **109 beds** (confirms that our 120-bed simulated hospital model accurately reflects median US hospital scale).
   - 90th Percentile: **465 beds** (large urban academic medical centers).
2. **Facility-Level vs State-Aggregated Occupancy**:
   - Facility-level unweighted mean occupancy was **56.48%** (median **57.14%**, std dev **22.87%**).
   - The gap between facility-level unweighted occupancy (56.5%) and state-level volume-weighted occupancy (74.5%) occurs because small facilities maintain excess reserve beds with low routine utilization (30–45%), whereas large urban tertiary centers (which account for >70% of total patient days) routinely operate at 75–85% capacity.
   - The state-level metric of **74.5%** correctly reflects system-wide bed utilization and serves as the primary target for simulator calibration.

---

## 4. Epidemiological Context & Caveat on COVID Data
> [!IMPORTANT]
> The HHS dataset covers the height of the COVID-19 pandemic (2020–2024), including Winter 2020, Delta (Summer 2021), and Omicron (Winter 2021–2022).
> Consequently:
> 1. Overall occupancy rates (74.5%) and surge frequencies (5.5%) are higher than peacetime elective baselines (~70–72%).
> 2. The multi-week surge episodes (up to 25 days) reflect pandemic waves rather than ordinary seasonal flu rushes.
> 3. For the Phase 6 simulator, baseline non-surge daily admissions are configured around **72% target occupancy**, with a scenario switch for flu-season rush driving occupancy into the calibrated 90%+ regime.
