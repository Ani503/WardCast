# WardCast Model Manifest & Architecture Card

This manifest documents all saved machine learning artifacts in `models/`, their clinical purpose, expected input fields, mathematical algorithms, and training provenance.

---

## Model 1: Length of Stay Classifier (`models/stay_model.joblib`)
- **Clinical Role**: Estimates individual patient length of stay across 11 discrete ordinal bands (`0-10` days to `More than 100 Days`) and outputs calibrated multi-class probability vectors.
- **Algorithm**: Artificial Neural Network with Backpropagation (`sklearn.neural_network.MLPClassifier`).
  - Architecture: 2 hidden layers (64, 32 units), ReLU activation, Adam optimizer, early stopping with validation monitoring.
  - L2 Regularization (alpha): `0.01`.
- **Preconditioning Pipeline**: `sklearn.compose.ColumnTransformer`
  - Nominal Categoricals: `OneHotEncoder(handle_unknown='ignore')`
  - Ordinal / Continuous: `SimpleImputer` (median/mode) + `StandardScaler`
- **Expected Input Dictionary / DataFrame**:
  - Clinical & Administrative Fields:
    - `Department` (str): e.g. `'gynecology'`, `'radiotherapy'`, `'anesthesia'`, `'surgery'`, `'TB & Chest disease'`
    - `Severity of Illness` (str): `'Minor'`, `'Moderate'`, `'Extreme'`
    - `Age` (str): Decade band e.g. `'31-40'`, `'51-60'`
    - `Type of Admission` (str): `'Emergency'`, `'Trauma'`, `'Urgent'`
    - `Available Extra Rooms in Hospital` (int): 0 to 24
    - `Visitors with Patient` (int): Accompanying visitors
    - `Admission_Deposit` (float): Financial deposit in USD ($1,800 to $11,000)
    - `Bed Grade` (float/int): 1.0, 2.0, 3.0, 4.0
    - `Hospital_code`, `Hospital_type_code`, `City_Code_Hospital`, `Hospital_region_code`, `Ward_Type`, `Ward_Facility_Code`, `City_Code_Patient`.
- **Output Contract**:
  - `predict()`: Discrete band index (0 to 10).
  - `predict_proba()`: 11-element array summing to 1.0 representing $[P(\text{Band}_0), \dots, P(\text{Band}_{10})]$.
- **Training Data**: Kaggle Healthcare Analytics II dataset (318,438 records). Cross-validated strictly with `GroupKFold(5)` grouped by `patientid`.

---

## Model 2: Patient Phenotype Clusterer (`models/clusterer.joblib`)
- **Clinical Role**: Assigns newly admitted inpatients to one of three validated clinical sub-populations for personalized ward routing and risk stratification.
- **Algorithm**: K-Means Clustering (`sklearn.cluster.KMeans`) at $k=3$.
  - Distance metric: Euclidean on 116-dimensional standardized feature space.
  - Benchmarked against Agglomerative Hierarchical (Ward) and BIRCH.
- **Identified Clusters**:
  - **Cluster 0**: *High-Acuity Surgical & Critical Inpatients* (Mean stay 33.6d, mean age 61.2 yrs, elevated severity).
  - **Cluster 1**: *Moderate-Stay General & Medical Inpatients* (Mean stay 31.7d, mean age 43.6 yrs).
  - **Cluster 2**: *Routine Elective & Rapid-Discharge Inpatients* (Mean stay 31.1d, mode 11–20d, lower acuity).
- **Expected Input**: Same feature schema as Model 1.
- **Output Contract**: Integer cluster ID ($c \in \{0, 1, 2\}$) and descriptive phenotype label.
- **Training Data**: Kaggle Healthcare Analytics II dataset (318,438 records).

---

## Model 3: Bed Surge Detector (`models/hmm_surge.joblib`)
- **Clinical Role**: Unsupervised early-warning bed capacity surge detector identifying latent hospital operational stress regimes before physical threshold exhaustion occurs.
- **Algorithm**: Continuous Gaussian Hidden Markov Model (`hmmlearn.hmm.GaussianHMM`) with full covariance matrices.
  - States: $k=3$ ordered regimes (`Normal`, `Busy / Alert`, `Surge / Critical`).
  - Decoding: Viterbi dynamic programming for optimal sequence estimation.
  - Risk Estimation: Forward-Backward algorithm for smooth posterior probabilities $P(S_t = \text{Surge} \mid \mathbf{X}_{1:t})$.
- **Expected Input**:
  - 2D observation vector: $[O_t, \Delta O_t]$
    - $O_t$: Inpatient occupancy rate ($0.0 \le O_t \le 1.2$).
    - $\Delta O_t = O_t - O_{t-1}$: Day-to-day census change.
- **Output Contract**:
  - `predict()`: Most likely hidden state ($0=\text{Normal}, 1=\text{Busy}, 2=\text{Surge}$).
  - `predict_proba()`: 3-element posterior probability vector $[P(\text{Normal}), P(\text{Busy}), P(\text{Surge})]$.
  - Transition probability matrix for multi-step ahead surge forecasting.
- **Training Data**: 766 contiguous chronological days from calibrated hospital simulation, validated on real-world HHS Arizona and Rhode Island capacity series.
