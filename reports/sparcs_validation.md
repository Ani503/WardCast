# WardCast Phase 4: SPARCS External Validation Report

## 1. Executive Summary: Transferability vs In-Domain Generalization
This report evaluates how well stay-prediction models generalize beyond their training domain, comparing:
1. **Direct Cross-Dataset Transfer**: Training on the Kaggle Healthcare Analytics dataset using harmonized clinical features (`Age`, `Admission Type`, `Severity`) and testing zero-shot on New York SPARCS 2014 inpatients.
2. **In-Domain Hospital-Held-Out Cross-Validation**: Training and testing strictly within SPARCS across 5 folds grouped by `Facility ID` (ensuring models are evaluated on completely unseen hospitals).
3. **Right-Censored Capping**: All stays of 120 days or longer ('120 +') are treated as right-censored and mapped into Band 10 ('More than 100 Days').

---

## 2. In-Domain Model Comparison on SPARCS (Hospital-Held-Out GroupKFold=5)
| Model Family | Accuracy | Delta vs Baseline (pp) | Macro-F1 | Recall Over 10 Days |
|---|---|---|---|---|
| **Majority Baseline** | **89.09%** | **0.00 pp** | **0.0856** | **0.00%** |
| **ANN (MLPClassifier)** | **89.05%** | **-0.04 pp** | **0.0877** | **1.67%** |
| **Decision Tree** | **89.02%** | **-0.07 pp** | **0.0949** | **8.41%** |
| **Random Forest** | **88.96%** | **-0.13 pp** | **0.1006** | **4.12%** |
| **Linear Regression** | **87.01%** | **-2.08 pp** | **0.1090** | **34.38%** |

---

## 3. Comparison of Pattern and Rankings Between Kaggle and SPARCS
1. **Model Hierarchy Concordance**:
   - On Kaggle, non-linear models (**ANN** at 42.49%) outperform simple decision trees, linear regression, and the majority baseline (27.48%).
   - On SPARCS, all models perform essentially at the majority baseline level (89.09%), dominated by the 89.09% short-stay class (ANN 89.05%, Decision Tree 89.02%, Random Forest 88.96%).
   - Macro-F1 averages across all 11 classes, ranging from 0.0856 (majority baseline) to 0.1090 (linear regression), reflecting that models capture only a small fraction of stays over 10 days (ANN: 1.67%, Decision Tree: 8.41%, Linear Regression: 34.38%).
2. **Transfer vs In-Domain**:
   - Zero-shot cross-dataset transfer collapses to **4.09%** because of the 89.09% short-stay base-rate in US acute care vs 7.4% in Kaggle.
   - Hospital-held-out validation confirms that models perform at majority-baseline level across unseen hospital facilities within the same health system, achieving 89.05% accuracy versus the 89.09% majority class baseline.

