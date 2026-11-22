# WardCast Cluster Profiles Report

## 1. Methodology & Model Comparison
Clustering was performed on 318,438 patient records across 116 preprocessed features.
Optimal cluster selection was confirmed using the Elbow method (inertia) and Silhouette analysis across k=2..7.

### Algorithm Comparison at k=3:
| Algorithm | Silhouette Score | Computational Complexity | Memory Footprint |
|---|---|---|---|
| **K-Means** | **0.0667** | O(N * k * d) — Fast, scalable | O(N) |
| **Agglomerative (Ward)** | **0.0761** | O(N^2 log N) — Prohibitive on 318k rows | O(N^2) |
| **BIRCH** | **0.0783** | O(N) — Tree-based streaming | O(tree size) |

---

## 2. Discovered Cluster Profiles

> **Weak Separation & Minimal Stay Variance**: Silhouette scores (0.0667 - 0.0783) and Davies-Bouldin indices (>3.1) confirm that patient records form weakly separated groups with high clinical overlap. The mean length of stay differences between clusters are under 3 days (Cluster A: 33.58 days, Cluster B: 31.66 days, Cluster C: 30.68 days; spread of 2.90 days). These clusters reflect administrative and demographic subgroups rather than distinct length-of-stay phenotypes.

### Cluster A
- **Population**: 102,107 patients (32.06%)
- **Length of Stay**: Geometric / Midpoint Mean = **33.58 days** (Most common band: `21-30`)
- **Demographics**: Mean Age = **61.2 years**, Average Visitors = **3.05**, Average Deposit = **$5,462.83**
- **Clinical Severity**:
|          |   Percentage (%) |
|:---------|-----------------:|
| Moderate |             61.8 |
| Minor    |             22.4 |
| Extreme  |             15.8 |
- **Top Hospital Departments**:
|              |   Percentage (%) |
|:-------------|-----------------:|
| gynecology   |             66.8 |
| anesthesia   |             16.5 |
| radiotherapy |             12.4 |
- **Admission Mode**:
|           |   Percentage (%) |
|:----------|-----------------:|
| Trauma    |             57.2 |
| Emergency |             30.1 |
| Urgent    |             12.7 |

---

### Cluster B
- **Population**: 114,320 patients (35.9%)
- **Length of Stay**: Geometric / Midpoint Mean = **31.66 days** (Most common band: `21-30`)
- **Demographics**: Mean Age = **43.6 years**, Average Visitors = **3.21**, Average Deposit = **$4,533.37**
- **Clinical Severity**:
|          |   Percentage (%) |
|:---------|-----------------:|
| Moderate |             64.1 |
| Extreme  |             33.9 |
| Minor    |              2   |
- **Top Hospital Departments**:
|              |   Percentage (%) |
|:-------------|-----------------:|
| gynecology   |             85.2 |
| radiotherapy |              8.1 |
| anesthesia   |              3.8 |
- **Admission Mode**:
|           |   Percentage (%) |
|:----------|-----------------:|
| Emergency |             43.5 |
| Trauma    |             42.2 |
| Urgent    |             14.3 |

---

### Cluster C
- **Population**: 102,011 patients (32.04%)
- **Length of Stay**: Geometric / Midpoint Mean = **30.68 days** (Most common band: `21-30`)
- **Demographics**: Mean Age = **41.5 years**, Average Visitors = **3.61**, Average Deposit = **$4,697.12**
- **Clinical Severity**:
|          |   Percentage (%) |
|:---------|-----------------:|
| Minor    |             59.6 |
| Moderate |             38.7 |
| Extreme  |              1.8 |
- **Top Hospital Departments**:
|              |   Percentage (%) |
|:-------------|-----------------:|
| gynecology   |             82.2 |
| anesthesia   |              8.2 |
| radiotherapy |              6.5 |
- **Admission Mode**:
|           |   Percentage (%) |
|:----------|-----------------:|
| Trauma    |             44.7 |
| Emergency |             36.5 |
| Urgent    |             18.8 |

---
