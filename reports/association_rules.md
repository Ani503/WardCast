# WardCast Association Rules Report: Prolonged Stay Drivers

## 1. Methodology & Algorithm Equivalence
Association rule mining was performed using both **Apriori** and **FP-Growth** algorithms via `mlxtend` with `min_support=0.03` and `min_confidence=0.20`.
- Frequent Itemsets Discovered: **2,752**
- Itemset Exact Match: **Confirmed Identical** between Apriori and FP-Growth.
- Target Consequent: **`Stay_Long (>50 Days)`**

---

## 2. Top 15 Rules by Lift (Support $\ge$ 1.5%)
| # | Antecedent Conditions | Support | Confidence | Lift | Clinical & Operational Implications |
|---|---|---|---|---|---|
| 1 | `Dept_Dept_gynecology, AdmType_Adm_Trauma, Visitors_Visitors_High_5+` | 0.0319 (3.19%) | 0.7015 (70.15%) | **3.59x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 2 | `AdmType_Adm_Trauma, Visitors_Visitors_High_5+` | 0.0424 (4.24%) | 0.7004 (70.04%) | **3.58x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 3 | `Severity_Severity_Moderate, Visitors_Visitors_High_5+` | 0.0467 (4.67%) | 0.6810 (68.10%) | **3.48x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 4 | `Dept_Dept_gynecology, Severity_Severity_Moderate, Visitors_Visitors_High_5+` | 0.0354 (3.54%) | 0.6790 (67.90%) | **3.47x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 5 | `Visitors_Visitors_High_5+` | 0.0833 (8.33%) | 0.6567 (65.67%) | **3.36x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 6 | `Age_Group_Middle_31-60, Visitors_Visitors_High_5+` | 0.0450 (4.50%) | 0.6559 (65.59%) | **3.35x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 7 | `Deposit_Deposit_Med_4-6k, Visitors_Visitors_High_5+` | 0.0402 (4.02%) | 0.6555 (65.55%) | **3.35x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 8 | `Dept_Dept_gynecology, Visitors_Visitors_High_5+` | 0.0638 (6.38%) | 0.6509 (65.09%) | **3.33x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 9 | `Dept_Dept_gynecology, Deposit_Deposit_Med_4-6k, Visitors_Visitors_High_5+` | 0.0304 (3.04%) | 0.6490 (64.90%) | **3.32x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 10 | `Dept_Dept_gynecology, Age_Group_Middle_31-60, Visitors_Visitors_High_5+` | 0.0351 (3.51%) | 0.6486 (64.86%) | **3.31x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 11 | `Rooms_Rooms_Med_3-4, Visitors_Visitors_High_5+` | 0.0457 (4.57%) | 0.6463 (64.63%) | **3.30x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 12 | `Deposit_Deposit_Low_<4k, Visitors_Visitors_High_5+` | 0.0322 (3.22%) | 0.6458 (64.58%) | **3.30x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 13 | `BedGrade_BedGrade_3, Visitors_Visitors_High_5+` | 0.0362 (3.62%) | 0.6448 (64.48%) | **3.30x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 14 | `Dept_Dept_gynecology, Rooms_Rooms_Med_3-4, Visitors_Visitors_High_5+` | 0.0352 (3.52%) | 0.6430 (64.30%) | **3.29x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
| 15 | `AdmType_Adm_Emergency, Visitors_Visitors_High_5+` | 0.0319 (3.19%) | 0.6191 (61.91%) | **3.16x** | Large visitor numbers correlate with high patient dependency and prolonged family involvement. |
