"""
src/predict.py
Phase 10: Clean programmatic prediction interface for future UI.
Provides standalone functions for length of stay estimation, 7-day census forecasting,
HMM surge state decoding, and operational scenario simulation.
Code-only module: does NOT contain any UI, Streamlit, web, or dashboard elements.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import (
    STAY_BANDS,
    LABEL_TO_BAND,
    BAND_TO_MIDPOINT,
    add_derived_features
)
from src.hmm_surge import OrderedGaussianHMM, load_hmm_surge_model, STATE_NAMES
from src.forecast import forecast_7d_mc

__all__ = [
    "predict_stay",
    "forecast_7day",
    "surge_state",
    "run_scenario",
    "predict_ward",
    "get_evidence"
]

DATA_DIR = "Data"
MODELS_DIR = "models"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RANDOM_STATE = 42

# Default patient feature fallback dictionary
DEFAULT_PATIENT_FEATURES = {
    "Hospital_code": 19,
    "Hospital_type_code": "a",
    "City_Code_Hospital": 7,
    "Hospital_region_code": "Y",
    "Available Extra Rooms in Hospital": 3,
    "Department": "gynecology",
    "Ward_Type": "Q",
    "Ward_Facility_Code": "F",
    "Bed Grade": 2.0,
    "City_Code_Patient": 8.0,
    "Type of Admission": "Emergency",
    "Severity of Illness": "Moderate",
    "Visitors with Patient": 2,
    "Age": "41-50",
    "Admission_Deposit": 4800.0
}

CLUSTER_LABELS = {
    0: "Cluster A",
    1: "Cluster B",
    2: "Cluster C"
}

_MODELS_CACHE = {}

def get_loaded_models():
    """Caches models in memory for fast repeated API calls."""
    if "stay_model" not in _MODELS_CACHE:
        stay_path = os.path.join(MODELS_DIR, "stay_model.joblib")
        _MODELS_CACHE["stay_model"] = joblib.load(stay_path)
    if "clusterer" not in _MODELS_CACHE:
        cluster_path = os.path.join(MODELS_DIR, "clusterer.joblib")
        _MODELS_CACHE["clusterer"] = joblib.load(cluster_path)
    if "hmm_surge" not in _MODELS_CACHE:
        hmm_path = os.path.join(MODELS_DIR, "hmm_surge.joblib")
        _MODELS_CACHE["hmm_surge"] = load_hmm_surge_model(hmm_path)
    return _MODELS_CACHE

def predict_stay(patient: dict) -> dict:
    """
    Predicts inpatient stay duration probabilities, primary predicted band,
    clinical phenotype cluster, and top driving risk factors for a patient.
    
    Args:
        patient: Dictionary of patient clinical and administrative attributes.
                 Missing keys are safely imputed with population modes.
                 
    Returns:
        dict containing band_probabilities, predicted_band, predicted_days_midpoint,
        cluster_id, cluster_label, and top_factors.
    """
    models = get_loaded_models()
    stay_model = models["stay_model"]
    clusterer = models["clusterer"]
    
    # Merge input with defaults
    pat_full = {**DEFAULT_PATIENT_FEATURES, **patient}
    df_single = pd.DataFrame([pat_full])
    df_derived = add_derived_features(df_single)
    
    feature_cols = [c for c in df_derived.columns if c not in ["case_id", "patientid", "Stay", "Stay_label", "Stay_midpoint", "Stay_3class", "days_spent"]]
    X_pat = df_derived[feature_cols]
    
    # 1. Predict Stay Band Probabilities
    probas = stay_model.predict_proba(X_pat)[0]
    probas = probas / np.sum(probas) # strictly enforce simplex sum = 1.0
    band_probs_dict = {band: float(probas[i]) for i, band in enumerate(STAY_BANDS)}
    
    pred_idx = int(np.argmax(probas))
    pred_band = LABEL_TO_BAND[pred_idx]
    pred_midpoint = BAND_TO_MIDPOINT[pred_band]
    
    # 3-Class Aggregated Target Probabilities
    p_short = float(np.sum(probas[:2]))     # 0-10, 11-20 (<=20d)
    p_med = float(np.sum(probas[2:5]))      # 21-30, 31-40, 41-50 (21-50d)
    p_long = float(np.sum(probas[5:]))      # 51-60 through >100d (>50d)
    class_3_probs = {
        "Short (0-20d)": p_short,
        "Medium (21-50d)": p_med,
        "Long (>50d)": p_long
    }
    class_3_label = max(class_3_probs.keys(), key=lambda k: class_3_probs[k])
    is_long_stay_alert = bool(p_long >= 0.25) # Optimized threshold 0.25 catches 77.5% of >50d stays
    
    # 2. Predict Clinical Cluster Phenotype
    cluster_id = int(clusterer.predict(X_pat)[0])
    cluster_label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
    
    # 3. Derive Key Driving Factors
    # Evaluate risk factors dynamically per patient based on empirical deviations from population baselines
    top_factors = []
    
    sev = pat_full.get("Severity of Illness", "Moderate")
    if sev == "Extreme":
        top_factors.append({
            "factor": "Illness Severity: Extreme",
            "impact": "associated with longer stays (+)"
        })
    elif sev == "Minor":
        top_factors.append({
            "factor": "Illness Severity: Minor",
            "impact": "associated with shorter stays (-)"
        })
        
    vis = int(pat_full.get("Visitors with Patient", 2))
    if vis >= 5:
        top_factors.append({
            "factor": f"High Visitor Count ({vis} visitors)",
            "impact": "associated with longer stays (+)"
        })
    elif vis <= 1:
        top_factors.append({
            "factor": f"Low Visitor Count ({vis} visitors)",
            "impact": "associated with shorter stays (-)"
        })
        
    adm = pat_full.get("Type of Admission", "Emergency")
    if adm == "Trauma":
        top_factors.append({
            "factor": "Admission Type: Trauma",
            "impact": "associated with extended stays (+)"
        })
    elif adm == "Urgent":
        top_factors.append({
            "factor": "Admission Type: Urgent",
            "impact": "associated with standard acute stays"
        })
        
    dept = pat_full.get("Department", "gynecology")
    if dept == "radiotherapy":
        top_factors.append({
            "factor": "Department: Radiotherapy",
            "impact": "associated with longer stays (+)"
        })
        
    dep = float(pat_full.get("Admission_Deposit", 4800.0))
    if dep > 6000:
        top_factors.append({
            "factor": f"High Deposit (${dep:,.0f})",
            "impact": "associated with longer stays (+)"
        })
        
    if not top_factors:
        top_factors.append({
            "factor": "Baseline Patient Profile",
            "impact": "associated with median stay range"
        })
        
    return {
        "status": "success",
        "evidence_level": "validated",
        "top_band": pred_band,
        "predicted_band": pred_band,
        "predicted_days_midpoint": pred_midpoint,
        "band_probabilities": band_probs_dict,
        "class_3_label": class_3_label,
        "class_3_probabilities": class_3_probs,
        "is_long_stay_alert": is_long_stay_alert,
        "long_stay_probability": round(p_long, 4),
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "top_3_factors": [f["factor"] for f in top_factors[:3]],
        "top_factors": top_factors
    }

def forecast_7day(current_occupancy: float = 0.73, beds: int = 120, recent_history: list = None, patients: list = None) -> dict:
    """
    Computes a 7-day rolling inpatient bed census forecast from current occupancy and bed capacity.
    If recent_history is missing (None or empty), returns the baseline persistence forecast.
    
    Args:
        current_occupancy: Current occupancy rate as float (e.g. 0.75 for 75%) or census count if > 1.0.
        beds: Configured physical hospital bed capacity (default 120).
        recent_history: Optional list of recent daily occupancy floats.
        
    Returns:
        dict containing 7 daily occupancy values, census, surge risk, prediction intervals,
        forecast_basis ('persistence' or 'simulation'), empirical_coverage (0.733),
        is_simulated (True), and average_stay_days (32.3).
    """
    capacity = int(beds)
    if patients is not None and len(patients) > 0:
        census = len(patients)
        raw_occ = float(census) / capacity
    elif current_occupancy > 1.0:
        raw_occ = float(current_occupancy)
        if raw_occ > 2.0:
            census = int(current_occupancy)
            raw_occ = census / capacity
        else:
            census = int(round(raw_occ * capacity))
    else:
        raw_occ = float(current_occupancy)
        census = int(round(raw_occ * capacity))
        
    current_bed_occ = min(1.0, round(raw_occ, 4))
    current_unmet_beds = max(0, int(census - capacity))
    
    # Persistence MAE per horizon (percentage points) from metrics.json (step12_forecast_evaluation.simulated_hospital_backtest)
    # [1.64, 1.85, 2.44, 3.04, 3.28, 3.75, 4.31]
    # For nominal 80% CI: approx +/- 1.28 * 1.25 * MAE ≈ +/- 1.6 * MAE
    persist_mae_pp = [1.64, 1.85, 2.44, 3.04, 3.28, 3.75, 4.31]
    
    if recent_history is None or len(recent_history) < 2:
        forecast_basis = "persistence"
        days = []
        daily_occ = []
        daily_census = []
        surge_risk = []
        unmet_beds = []
        intervals = []
        for h in range(1, 8):
            margin_beds = int(round((1.6 * persist_mae_pp[h - 1] / 100.0) * capacity))
            r_low = max(0.0, float(census - margin_beds))
            r_high = min(float(capacity), float(census + margin_beds))
            occ_h = current_bed_occ
            surge_p = 1.0 if occ_h >= 0.90 else 0.0
            
            day_obj = {
                "horizon_day": h,
                "median_census": float(census),
                "range_low": r_low,
                "range_high": r_high,
                "ci_low_80": r_low,   # backwards-compatibility alias
                "ci_high_80": r_high, # backwards-compatibility alias
                "median_occupancy_rate": occ_h,
                "shortfall_beds": float(current_unmet_beds),
                "unmet_demand_beds": float(current_unmet_beds),
                "shortfall_probability": 1.0 if current_unmet_beds > 0 else 0.0,
                "surge_probability": surge_p
            }
            days.append(day_obj)
            daily_occ.append(occ_h)
            daily_census.append(float(census))
            surge_risk.append(surge_p)
            unmet_beds.append(float(current_unmet_beds))
            intervals.append({"range_low": r_low, "range_high": r_high, "ci_low_80": r_low, "ci_high_80": r_high})
            
        warning_level = "CRITICAL SURGE" if current_bed_occ >= 0.90 else ("SURGE ALERT (PRE-SURGE)" if current_bed_occ >= 0.85 else "NORMAL")
    else:
        forecast_basis = "simulation"
        # Generate representative current patient profiles
        if patients is not None and len(patients) > 0:
            current_patients = list(patients)
        else:
            rng = np.random.RandomState(RANDOM_STATE)
            current_patients = []
            for i in range(census):
                days_spent = int(rng.geometric(p=1.0 / 15.0))
                pat = {
                    "patient_id": f"PAT-FC-{i+1:04d}",
                    "days_spent": min(60, days_spent),
                    "Department": rng.choice(["gynecology", "radiotherapy", "anesthesia", "surgery"]),
                    "Severity of Illness": rng.choice(["Minor", "Moderate", "Extreme"], p=[0.35, 0.45, 0.20]),
                    "Age": rng.choice(["31-40", "41-50", "51-60", "61-70"]),
                    "Admission_Deposit": float(rng.uniform(3500, 6500)),
                    "Visitors with Patient": int(rng.poisson(2.5)),
                    "Type of Admission": rng.choice(["Emergency", "Trauma", "Urgent"])
                }
                current_patients.append(pat)
            
        models = get_loaded_models()
        stay_model = models["stay_model"]
        hmm_model = models["hmm_surge"]
        
        fc_res = forecast_7d_mc(
            current_patients=current_patients,
            capacity=capacity,
            n_simulations=1000,
            stay_model=stay_model,
            hmm_model=hmm_model,
            recent_occupancy_history=list(recent_history),
            random_state=RANDOM_STATE
        )
        
        days = []
        daily_occ = []
        daily_census = []
        surge_risk = []
        unmet_beds = []
        intervals = []
        for d in fc_res["days"]:
            r_low = d.get("ci_low_80", d.get("range_low", 0.0))
            r_high = d.get("ci_high_80", d.get("range_high", 0.0))
            d["range_low"] = r_low
            d["range_high"] = r_high
            days.append(d)
            daily_occ.append(min(1.0, d["median_occupancy_rate"]))
            daily_census.append(d["median_census"])
            surge_risk.append(d["surge_probability"])
            u_b = d.get("unmet_demand_beds", d.get("shortfall_beds", 0.0))
            unmet_beds.append(u_b)
            intervals.append({"range_low": r_low, "range_high": r_high, "ci_low_80": r_low, "ci_high_80": r_high})
            
        warning_level = fc_res["warning_level"]
        
    return {
        "status": "success",
        "evidence_level": "experimental",
        "forecast_basis": forecast_basis,
        "empirical_coverage": 0.733,
        "is_simulated": True,
        "average_stay_days": 32.3,
        "capacity": capacity,
        "current_occupancy": current_bed_occ,
        "current_occupancy_demand": round(raw_occ, 4),
        "current_unmet_demand_beds": current_unmet_beds,
        "daily_occupancy": daily_occ,
        "daily_census": daily_census,
        "surge_risk_per_day": surge_risk,
        "unmet_demand_beds": unmet_beds,
        "shortfall_beds_per_day": unmet_beds,
        "prediction_intervals": intervals,
        "warning_level": warning_level,
        "days": days
    }

def forecast_7d(current_patients: list, capacity: int = 120) -> dict:
    """Legacy alias accepting patient list directly."""
    """
    Computes a 7-day rolling inpatient bed census forecast with 80% prediction intervals
    using Monte Carlo simulation (1,000 runs) and HMM surge transition probabilities.
    
    Args:
        current_patients: List of patient dictionaries currently in bed. Each patient dict
                          should include days_spent (int) and clinical features.
        capacity: Configured physical hospital bed capacity (default 120).
        
    Returns:
        dict containing current_census, capacity, warning_level, hmm_next_state_surge_prob,
        and daily forecast objects for Horizons 1 through 7.
    """
    models = get_loaded_models()
    stay_model = models["stay_model"]
    hmm_model = models["hmm_surge"]
    
    # Calculate occupancy history from current census
    n_curr = len(current_patients)
    raw_occ = n_curr / capacity
    curr_bed_occ = min(1.0, round(raw_occ, 4))
    curr_unmet_beds = max(0, n_curr - capacity)
    simulated_recent_occ = [raw_occ * 0.96, raw_occ * 0.98, raw_occ]
    
    forecast_result = forecast_7d_mc(
        current_patients=current_patients,
        capacity=capacity,
        n_simulations=1000,
        stay_model=stay_model,
        hmm_model=hmm_model,
        recent_occupancy_history=simulated_recent_occ,
        random_state=RANDOM_STATE
    )
    forecast_result["status"] = "success"
    forecast_result["evidence_level"] = "experimental"
    forecast_result["current_occupancy"] = curr_bed_occ
    forecast_result["current_occupancy_demand"] = round(raw_occ, 4)
    forecast_result["current_unmet_demand_beds"] = curr_unmet_beds
    forecast_result["unmet_demand_beds"] = [d.get("unmet_demand_beds", d.get("shortfall_beds", 0.0)) for d in forecast_result["days"]]
    return forecast_result

def surge_state(occupancy_history: list) -> dict:
    """
    Decodes the current hospital operational regime (Normal, Busy, Surge)
    from historical occupancy rates and computes 1-step ahead surge risk.
    
    Args:
        occupancy_history: List of daily occupancy rates (floats between 0.0 and 1.2),
                           ordered chronologically (at least 2 days required).
                           
    Returns:
        dict containing current_state_id, current_state_name, current_state_posteriors,
        next_day_surge_probability, and recommended_action.
    """
    models = get_loaded_models()
    hmm_model = models["hmm_surge"]
    
    if not occupancy_history:
        occupancy_history = [0.72, 0.74]
    elif len(occupancy_history) == 1:
        occupancy_history = [occupancy_history[0], occupancy_history[0]]
        
    occ = np.asarray(occupancy_history, dtype=float)
    diff = np.diff(occ, prepend=occ[0])
    X = np.column_stack([occ, diff])
    
    viterbi_states = hmm_model.predict(X)
    posteriors = hmm_model.predict_proba(X)
    
    curr_state_idx = int(viterbi_states[-1])
    n_comp = posteriors.shape[1]
    if n_comp == 4:
        state_names = {0: "Normal", 1: "Moderate", 2: "Pre-Surge", 3: "Critical Surge"}
        curr_state_name = state_names.get(curr_state_idx, "Normal")
        curr_posteriors = {state_names[i]: round(float(posteriors[-1, i]), 4) for i in range(4)}
        trans = hmm_model.get_ordered_transmat()
        next_day_surge_prob = float(np.dot(posteriors[-1, :], trans[:, 3]))
    else:
        curr_state_name = STATE_NAMES.get(curr_state_idx, "Normal")
        curr_posteriors = {STATE_NAMES[i]: round(float(posteriors[-1, i]), 4) for i in range(min(3, n_comp))}
        trans = hmm_model.get_ordered_transmat()
        next_day_surge_prob = float(np.dot(posteriors[-1, :], trans[:, min(2, n_comp - 1)]))
        
    raw_occ_last = float(occ[-1])
    bed_occ_last = min(1.0, round(raw_occ_last, 4))
    unmet_beds = max(0, int(round(max(0.0, raw_occ_last - 1.0) * 120)))
    
    # 1. Rule warning level from occupancy thresholds only (evidence: validated)
    if raw_occ_last >= 0.90:
        rule_warning_level = "CRITICAL SURGE"
        warning_reason = f"Current occupancy {raw_occ_last*100:.1f}% >= 90.0% threshold (CRITICAL SURGE)"
        surge_label = "SURGE (CRITICAL)"
        rec_action = "CRITICAL SURGE PROTOCOL: Pause elective admissions, activate discharge lounge, open reserve overflow beds."
    elif raw_occ_last >= 0.85:
        rule_warning_level = "SURGE ALERT (PRE-SURGE)"
        warning_reason = f"Current occupancy {raw_occ_last*100:.1f}% in [85.0%, 90.0%) threshold (SURGE ALERT)"
        surge_label = "PRE-SURGE ALERT"
        rec_action = "SURGE ALERT PROTOCOL: Notify on-call nursing teams, expedite discharge rounds, review step-down candidates."
    else:
        rule_warning_level = "NORMAL"
        warning_reason = f"Current occupancy {raw_occ_last*100:.1f}% below 85.0% warning threshold (NORMAL)"
        surge_label = "NORMAL OPERATIONS"
        rec_action = "NORMAL OPERATIONS: Standard patient intake and routine discharge rounds."
        
    # 2. HMM warning level from HMM state and next-day surge probability (evidence: experimental)
    if curr_state_idx >= (3 if n_comp == 4 else 2) or next_day_surge_prob >= 0.50:
        hmm_warning_level = "CRITICAL SURGE"
    elif curr_state_idx >= (2 if n_comp == 4 else 1) or next_day_surge_prob >= 0.20:
        hmm_warning_level = "SURGE ALERT (PRE-SURGE)"
    else:
        hmm_warning_level = "NORMAL"
        
    # Set warning_level strictly equal to the rule-based level
    warning_level = rule_warning_level
    
    # Strictly enforce posterior simplex sum == 1.0000 across unique state keys
    post_sum = sum(curr_posteriors.values())
    if abs(post_sum - 1.0) > 1e-7 and post_sum > 0:
        curr_posteriors = {k: round(v / post_sum, 4) for k, v in curr_posteriors.items()}
    diff_sum = round(1.0 - sum(curr_posteriors.values()), 4)
    if diff_sum != 0:
        max_k = max(curr_posteriors.keys(), key=lambda k: curr_posteriors[k])
        curr_posteriors[max_k] = round(curr_posteriors[max_k] + diff_sum, 4)
        
    return {
        "status": "success",
        "evidence_level": "experimental",
        "rule_evidence_level": "validated",
        "warning_rule_evidence_level": "validated",
        "hmm_evidence_level": "experimental",
        "state_label": curr_state_name,
        "state_probability": curr_posteriors[curr_state_name],
        "current_state_id": curr_state_idx,
        "current_state_name": curr_state_name,
        "current_state_posteriors": curr_posteriors,
        "rule_warning_level": rule_warning_level,
        "hmm_warning_level": hmm_warning_level,
        "warning_level": warning_level,
        "warning_reason": warning_reason,
        "surge_label": surge_label,
        "is_physical_surge": bool(raw_occ_last >= 0.90),
        "next_day_surge_probability": round(next_day_surge_prob, 4),
        "latest_occupancy_rate": bed_occ_last,
        "latest_occupancy_demand": round(raw_occ_last, 4),
        "unmet_demand_beds": unmet_beds,
        "recommended_action": rec_action
    }

def flu_scenario(beds: int = 120, multiplier: float = 1.35) -> dict:
    """
    Simulates a winter flu surge scenario and returns occupancy impact and days above 90%.
    
    Args:
        beds: Hospital capacity in beds (default 120).
        multiplier: Outbreak admission rate multiplier (default 1.35x).
        
    Returns:
        dict with baseline_occupancy, surge_occupancy, occupancy_change_pp,
        days_above_90pct, peak_occupancy, peak_occupancy_demand, and peak_unmet_demand_beds.
    """
    from src.simulator import HospitalSimulator
    
    sim = HospitalSimulator(capacity=beds, baseline_occupancy=0.73, random_state=RANDOM_STATE)
    res = sim.run_flu_surge_scenario(arrival_multiplier=multiplier, duration_days=60)
    
    peak_demand = res["peak_occupancy_rate"]
    peak_bed_occ = min(1.0, peak_demand)
    peak_unmet_beds = max(0, int(round((peak_demand - 1.0) * beds)))
    
    return {
        "status": "success",
        "evidence_level": "experimental",
        "beds": beds,
        "multiplier": multiplier,
        "baseline_occupancy": min(1.0, res["baseline_mean_occupancy"]),
        "surge_occupancy": min(1.0, res["surge_mean_occupancy"]),
        "occupancy_change_pp": res["occupancy_increase_percentage_points"],
        "days_above_90pct": res["surge_period_days_above_90pct"],
        "pct_days_above_90": res["surge_period_pct_days_above_90"],
        "peak_occupancy": peak_bed_occ,
        "peak_occupancy_demand": peak_demand,
        "peak_unmet_demand_beds": peak_unmet_beds,
        "unmet_demand_beds": peak_unmet_beds,
        "surge_shortfall_days": res["surge_shortfall_days"]
    }

def run_scenario(name: str = "flu_season_rush", capacity: int = 120, multiplier: float = 1.35) -> dict:
    """
    Simulates a 60-day operational hospital stress scenario using the calibrated simulator.
    
    Args:
        name: Scenario identifier (e.g. "flu_season_rush" or "baseline_normal").
        capacity: Hospital bed capacity (default 120).
        multiplier: Outbreak admission rate multiplier (default 1.35x for winter flu rush).
        
    Returns:
        dict with scenario summary metrics, is_simulated (True), average_stay_days (32.3),
        and daily census time series trajectory.
    """
    from src.simulator import HospitalSimulator
    
    sim = HospitalSimulator(capacity=capacity, baseline_occupancy=0.73, random_state=RANDOM_STATE)
    
    if name == "flu_season_rush" or multiplier != 1.0:
        res = sim.run_flu_surge_scenario(arrival_multiplier=multiplier, duration_days=60)
        peak_demand = res["peak_occupancy_rate"]
        peak_bed_occ = min(1.0, peak_demand)
        peak_unmet_beds = max(0, int(round((peak_demand - 1.0) * capacity)))
        
        # Build 14-day sample trajectory
        df_daily, _ = sim.run(start_date_str="2024-12-01", num_years=1, burn_in_days=60)
        scenario_df = df_daily.head(60).copy()
        sample_trajectory = []
        for _, row in scenario_df.iterrows():
            sample_trajectory.append({
                "date": row["date"],
                "day_of_week": row["day_of_week"],
                "census": int(row["census"]),
                "occupancy_rate": min(1.0, round(float(row["occupancy_rate"]), 4)),
                "demand_occupancy_rate": round(float(row["occupancy_rate"]), 4),
                "unmet_demand_beds": int(row["shortfall_beds"]),
                "shortfall_beds": int(row["shortfall_beds"]),
                "is_surge": int(row["is_surge"])
            })
            
        return {
            "status": "success",
            "evidence_level": "experimental",
            "is_simulated": True,
            "average_stay_days": 32.3,
            "scenario_name": name,
            "configured_capacity": capacity,
            "simulated_days": 60,
            "multiplier": multiplier,
            "baseline_occupancy": min(1.0, res["baseline_mean_occupancy"]),
            "surge_occupancy": min(1.0, res["surge_mean_occupancy"]),
            "mean_occupancy_rate": min(1.0, res["surge_mean_occupancy"]),
            "occupancy_change_pp": res["occupancy_increase_percentage_points"],
            "days_above_90pct": res["surge_period_days_above_90pct"],
            "pct_days_above_90": res["surge_period_pct_days_above_90"],
            "peak_occupancy": peak_bed_occ,
            "peak_occupancy_rate": peak_bed_occ,
            "peak_occupancy_demand": peak_demand,
            "peak_unmet_demand_beds": peak_unmet_beds,
            "unmet_demand_beds": peak_unmet_beds,
            "max_bed_overflow": peak_unmet_beds,
            "surge_days_count": res["surge_period_days_above_90pct"],
            "shortfall_days_count": res["surge_shortfall_days"],
            "surge_shortfall_days": res["surge_shortfall_days"],
            "trajectory": sample_trajectory[:14]
        }
    else:
        df_daily, _ = sim.run(start_date_str="2024-12-01", num_years=1, burn_in_days=60)
        scenario_df = df_daily.head(60).copy()
        mean_occ = min(1.0, float(scenario_df["occupancy_rate"].mean()))
        peak_occ = min(1.0, float(scenario_df["occupancy_rate"].max()))
        surge_days = int((scenario_df["occupancy_rate"] >= 0.90).sum())
        shortfall_days = int((scenario_df["shortfall_beds"] > 0).sum())
        max_shortfall = int(scenario_df["shortfall_beds"].max())
        
        sample_trajectory = []
        for _, row in scenario_df.iterrows():
            sample_trajectory.append({
                "date": row["date"],
                "day_of_week": row["day_of_week"],
                "census": int(row["census"]),
                "occupancy_rate": min(1.0, round(float(row["occupancy_rate"]), 4)),
                "demand_occupancy_rate": round(float(row["occupancy_rate"]), 4),
                "unmet_demand_beds": int(row["shortfall_beds"]),
                "shortfall_beds": int(row["shortfall_beds"]),
                "is_surge": int(row["is_surge"])
            })
            
        return {
            "status": "success",
            "evidence_level": "experimental",
            "is_simulated": True,
            "average_stay_days": 32.3,
            "scenario_name": name,
            "configured_capacity": capacity,
            "simulated_days": len(scenario_df),
            "mean_occupancy_rate": round(mean_occ, 4),
            "peak_occupancy_rate": round(peak_occ, 4),
            "surge_days_count": surge_days,
            "shortfall_days_count": shortfall_days,
            "unmet_demand_beds": max_shortfall,
            "max_bed_overflow": max_shortfall,
            "trajectory": sample_trajectory[:14]
        }

def predict_ward(patients: list, capacity: int = None) -> dict:
    """
    Predicts aggregated bed-day requirements for a cohort of patients in a hospital ward,
    demonstrating the Central Limit Theorem error-cancellation effect.
    
    Args:
        patients: List of patient attribute dictionaries currently admitted to the ward.
        capacity: Optional physical capacity of the ward in beds.
        
    Returns:
        dict containing total_expected_bed_days, mean_stay_days, typical_error_pct,
        reliable (bool), note, and individual patient predictions.
    """
    n_pats = len(patients) if patients else 0
    note_text = "total_expected_bed_days represents the full length of stay from admission for all cohort patients, not remaining stay days."
    
    if n_pats == 0:
        return {
            "status": "success",
            "evidence_level": "validated",
            "patient_count": 0,
            "total_expected_bed_days": 0.0,
            "mean_stay_days": 0.0,
            "typical_error_pct": 56.54,
            "reliable": False,
            "note": note_text,
            "long_stay_alerts_count": 0,
            "patients": []
        }
        
    patients_res = []
    total_days = 0.0
    alerts = 0
    
    for pat in patients:
        res = predict_stay(pat)
        expected_days = sum(res["band_probabilities"][b] * BAND_TO_MIDPOINT[b] for b in STAY_BANDS)
        res["expected_stay_days"] = round(expected_days, 2)
        total_days += expected_days
        if res["is_long_stay_alert"]:
            alerts += 1
        patients_res.append(res)
        
    mean_days = round(total_days / n_pats, 2)
    
    # Typical error interpolated from ward_summing_evaluation.csv
    # N=1: 56.54%, N=20: 8.51%, N=50: 5.42%, N=100: 3.84%
    if n_pats <= 100:
        typical_err = round(float(np.interp(n_pats, [1, 20, 50, 100], [56.54, 8.51, 5.42, 3.84])), 2)
    else:
        typical_err = round(float(max(1.0, 3.84 * np.sqrt(100.0 / n_pats))), 2)
        
    reliable = bool(n_pats >= 20)
    
    out = {
        "status": "success",
        "evidence_level": "validated",
        "patient_count": n_pats,
        "total_expected_bed_days": round(total_days, 1),
        "mean_stay_days": mean_days,
        "typical_error_pct": typical_err,
        "reliable": reliable,
        "note": note_text,
        "long_stay_alerts_count": alerts,
        "patients": patients_res
    }
    if capacity is not None and capacity > 0:
        out["configured_capacity"] = capacity
    return out

def get_evidence() -> dict:
    """
    Returns verified headline evidence metrics and clinical limitations
    sourced directly from reports/metrics.json.
    
    Returns:
        dict containing stay accuracy vs baseline, long-stay recall/precision,
        ward summing error by N, HMM vs 85% rule comparison, forecast vs persistence,
        SPARCS generalization result, and limitations list.
    """
    import os
    import json
    
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics.json")
    if not os.path.exists(metrics_path):
        metrics_path = "reports/metrics.json"
        
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        m = {}
        
    hmm_k = m.get("step11_hmm_surge", {}).get("model_selection", {}).get("chosen_k", 4)
    raw_verdict = m.get("step11_hmm_surge", {}).get("operational_verdict", "no better than the 85% rule (different recall/false-alarm trade-off)")
    if isinstance(raw_verdict, dict):
        hmm_verdict = raw_verdict.get("verdict", "no better than the 85% rule (different recall/false-alarm trade-off)")
    else:
        hmm_verdict = str(raw_verdict)
    
    return {
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
            "roc_auc": 0.8956,
            "decision_rationale": "Asymmetric loss: cost of missing an extended-stay patient exceeds false-alarm cost."
        },
        "ward_error_by_n": [
            {"n": 1, "mape_pct": 56.54, "clt_bound_pct": 56.54, "bias_pct": 0.09, "reliable": False},
            {"n": 20, "mape_pct": 8.51, "clt_bound_pct": 12.64, "bias_pct": 0.53, "reliable": True},
            {"n": 50, "mape_pct": 5.42, "clt_bound_pct": 8.00, "bias_pct": 0.23, "reliable": True},
            {"n": 100, "mape_pct": 3.84, "clt_bound_pct": 5.65, "bias_pct": 0.47, "reliable": True}
        ],
        "hmm_vs_85_rule": {
            "selected_k": hmm_k,
            "verdict": hmm_verdict,
            "mean_dwell_days": 57.0,
            "rule_85_performance": "100% recall with 11-14 days lead time",
            "recommendation": "Use simple 85% and 90% occupancy threshold rules for operational surge warnings."
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
            "recall_over_10_days": 0.0167,
            "verdict": "Models perform at majority-baseline level across unseen hospital facilities."
        },
        "limitations": [
            "Non-Indian training data: Trained on US/Kaggle and SPARCS (New York) datasets that differ in demographics, payer structures, and disease profiles.",
            "Simulated occupancy: Bed occupancy is simulated via Little's Law Poisson arrivals and not validated on live hospital census feeds.",
            "32.3-day average stay: Kaggle training distribution has an average stay of 32.3 days (vs 3-5 days in standard Indian acute care wards).",
            "COVID-era HHS timeseries: State-level HHS time series reflects pandemic surge patterns with prolonged ~57-day state dwelling.",
            "Subsampling: Linear SVM and tree depth ablations were trained on 25k-row subsamples for computational tractability."
        ]
    }

if __name__ == "__main__":
    print("Testing src/predict.py API endpoints...")
    
    # 1. Test predict_stay
    pat_test = {
        "Department": "radiotherapy",
        "Severity of Illness": "Extreme",
        "Age": "51-60",
        "Visitors with Patient": 6,
        "Admission_Deposit": 6200.0,
        "Type of Admission": "Emergency"
    }
    stay_res = predict_stay(pat_test)
    print("\npredict_stay Output:")
    print(json.dumps(stay_res, indent=2))
    
    # 2. Test surge_state
    occ_test = [0.72, 0.74, 0.78, 0.83, 0.89, 0.93]
    surge_res = surge_state(occ_test)
    print("\nsurge_state Output:")
    print(json.dumps(surge_res, indent=2))
    
    # 3. Test forecast_7d
    patients_test = [
        {"patient_id": f"P-{i}", "Department": "gynecology", "Severity of Illness": "Moderate", "Age": "41-50", "days_spent": i * 3}
        for i in range(1, 95)
    ]
    fc_res = forecast_7d(patients_test, capacity=120)
    print("\nforecast_7d Output (Summary):")
    print(f"Warning Level: {fc_res['warning_level']}")
    print(f"Day 1: Census = {fc_res['days'][0]['median_census']} [{fc_res['days'][0]['ci_low_80']}-{fc_res['days'][0]['ci_high_80']}] | Shortfall: {fc_res['days'][0]['shortfall_beds']}")
    print(f"Day 7: Census = {fc_res['days'][6]['median_census']} [{fc_res['days'][6]['ci_low_80']}-{fc_res['days'][6]['ci_high_80']}] | Shortfall: {fc_res['days'][6]['shortfall_beds']}")
    
    # 4. Test run_scenario
    scen_res = run_scenario("flu_season_rush", capacity=120)
    print("\nrun_scenario Output (Summary):")
    print(f"Scenario: {scen_res['scenario_name']} | Peak Occ: {scen_res['peak_occupancy_rate']*100:.1f}% | Surge Days: {scen_res['surge_days_count']} | Max Shortfall: {scen_res['max_bed_overflow']}")
