#!/usr/bin/env python3
"""
run_demo.py - WardCast Machine Learning Core Demonstration
University Project: PCCAIML 502 (Introduction to Machine Learning)

Demonstrates the 4 operational inference functions from src/predict.py:
  1. predict_stay(sample_patient)
  2. forecast_7day(current_occupancy=0.78, beds=120)
  3. surge_state(recent_occupancy_history)
  4. flu_scenario(beds=120, multiplier=1.35)

Outputs clean, human-readable terminal text with zero raw dict dumps.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import (
    predict_stay,
    forecast_7day,
    surge_state,
    flu_scenario,
    run_scenario,
    predict_ward,
    get_evidence
)

def print_separator(title=""):
    width = 78
    if title:
        padding = max(0, width - len(title) - 4)
        left = padding // 2
        right = padding - left
        print(f"\n{'=' * left}  {title.upper()}  {'=' * right}")
    else:
        print(f"{'=' * width}")

def demo_predict_stay():
    print_separator("Demo 1: Inpatient Length of Stay Prediction")
    print("Scenario: A 55-year-old trauma patient admitted via Emergency to Radiotherapy.")
    
    sample_patient = {
        "Department": "radiotherapy",
        "Severity of Illness": "Extreme",
        "Age": "51-60",
        "Visitors with Patient": 6,
        "Admission_Deposit": 6200.0,
        "Type of Admission": "Emergency",
        "Bed Grade": 3.0
    }
    
    result = predict_stay(sample_patient)
    
    print("\n[Input Patient Characteristics]")
    for k, v in sample_patient.items():
        print(f"  - {k:<25}: {v}")
        
    print("\n[Model Inference & Point Estimates]")
    print(f"  Primary Predicted Band      : {result['predicted_band']} Days (midpoint ~{result['predicted_days_midpoint']:.1f} days)")
    print(f"  3-Class Triage Category     : {result['class_3_label']}")
    print(f"  Phenotypic Patient Cluster  : {result['cluster_label']}")
    
    print("\n[Out-of-Fold 11-Band Calibrated Probabilities]")
    print("  " + "-" * 42)
    print("  | Stay Duration Band | Probability       |")
    print("  " + "-" * 42)
    for band, prob in result["band_probabilities"].items():
        bar = "#" * int(round(prob * 40))
        print(f"  | {band:<18} | {prob*100:>5.1f}%  {bar:<12} |")
    print("  " + "-" * 42)
    
    print("\n[Clinical Decision Support: Long-Stay Alert Engine (theta = 0.25)]")
    p_long = result["long_stay_probability"]
    alert = result["is_long_stay_alert"]
    alert_str = "ACTIVE (TRIGGERED)" if alert else "INACTIVE (NORMAL)"
    print(f"  Estimated Probability of Stay > 50 Days : {p_long*100:.1f}%")
    print(f"  Long-Stay High-Acuity Alert Status       : {alert_str}")
    if alert:
        print("  Operational Recommendation               : Flag for early multidisciplinary discharge")
        print("                                             planning and step-down rehab assessment.")
        
    print("\n[Top Risk Factors (Statistical Associations)]")
    for idx, factor in enumerate(result["top_factors"][:3], 1):
        print(f"  {idx}. {factor['factor']} -> {factor['impact']}")

def demo_forecast_7day():
    print_separator("Demo 2: 7-Day Rolling Bed Census Forecast")
    print("Scenario: Hospital has 120 licensed beds currently operating at 78.0% occupancy (94 patients).")
    
    fc = forecast_7day(current_occupancy=0.78, beds=120)
    
    print(f"\n[Baseline Status]")
    print(f"  Licensed Bed Capacity : {fc['capacity']} beds")
    print(f"  Current Occupancy     : {fc['current_occupancy']*100:.1f}% ({int(round(fc['current_occupancy']*fc['capacity']))} occupied)")
    print(f"  Forecast Basis        : {fc['forecast_basis'].capitalize()} (Held-out coverage: {fc['empirical_coverage']*100:.1f}%)")
    print(f"  Forecast Warning Tier : {fc['warning_level']}")
    
    print("\n[7-Day Census Trajectory & 80% Prediction Intervals]")
    print("  " + "-" * 72)
    print("  | Horizon | Expected Census | 80% CI Range | Occupancy (%) | Surge Risk (>=90%) |")
    print("  " + "-" * 72)
    for day in fc["days"]:
        h = day["horizon_day"]
        census = day["median_census"]
        low = day.get("range_low", day.get("ci_low_80"))
        high = day.get("range_high", day.get("ci_high_80"))
        occ = day["median_occupancy_rate"] * 100
        surge = day["surge_probability"] * 100
        print(f"  | Day +{h}  | {census:>5.1f} beds   | [{low:>4.1f} - {high:>4.1f}] |    {occ:>5.1f}%    |       {surge:>5.1f}%        |")
    print("  " + "-" * 72)
    print("  Note: 80% prediction intervals achieve 73.3% empirical coverage on held-out test window (SE 8.1%).")

def demo_surge_state():
    print_separator("Demo 3: HMM Operational Regime Decoding & Surge Risk")
    print("Scenario: Tracking recent 6-day occupancy trajectory: [74%, 78%, 82%, 86%, 89%, 92%].")
    
    occ_hist = [0.74, 0.78, 0.82, 0.86, 0.89, 0.92]
    result = surge_state(occ_hist)
    
    print("\n[Recent Occupancy History]")
    print("  " + " -> ".join([f"{x*100:.1f}%" for x in occ_hist]))
    
    print("\n[HMM Regime Decoder (k=4 Ordered States)]")
    print(f"  Latest Occupancy Rate      : {result['latest_occupancy_rate']*100:.1f}% ({'PHYSICAL SURGE >=90%' if result['latest_occupancy_rate'] >= 0.90 else 'Below 90%'})")
    print(f"  Decoded Hidden State       : State {result['current_state_id']} ({result['current_state_name']})")
    print(f"  State Confidence           : {result['state_probability']*100:.1f}%")
    print(f"  1-Day Ahead Surge Risk     : {result['next_day_surge_probability']*100:.1f}%")
    print(f"  Rule Warning Tier (Valid.) : {result['rule_warning_level']}")
    print(f"  HMM Warning Tier (Experim.): {result['hmm_warning_level']}")
    print(f"  Warning Reason             : {result['warning_reason']}")
    
    print("\n[State Posterior Distribution (k=4 States, Sum = 100%)]")
    for state, prob in result["current_state_posteriors"].items():
        bar = "=" * int(round(prob * 30))
        print(f"  - State: {state:<15} : {prob*100:>5.1f}%  {bar}")
        
    print("\n[Recommended Operational Protocol]")
    print(f"  >>> {result['recommended_action']}")

def demo_flu_scenario():
    print_separator("Demo 4: Seasonal Flu Epidemic Stress Test")
    print("Scenario: Assessing resilience under a 60-day winter epidemic rush (+35% admissions).")
    
    result = flu_scenario(beds=120, multiplier=1.35)
    
    print("\n[Hospital Configuration & Stress Parameters]")
    print(f"  Bed Capacity               : {result['beds']} beds")
    print(f"  Admission Rate Multiplier  : {result['multiplier']:.2f}x baseline intake")
    print(f"  Simulation Window          : 60 days (Winter season)")
    
    print("\n[Stress Test Simulated Outcomes]")
    print(f"  Baseline Mean Occupancy    : {result['baseline_occupancy']*100:.1f}%")
    print(f"  Surge Mean Occupancy       : {result['surge_occupancy']*100:.1f}%")
    print(f"  Net Occupancy Increase     : +{result['occupancy_change_pp']:.2f} percentage points")
    print(f"  Peak Bed Occupancy         : {result['peak_occupancy']*100:.1f}% (All {result['beds']} licensed beds occupied)")
    print(f"  Peak Unmet Patient Demand  : {result['peak_unmet_demand_beds']} beds ({result['peak_occupancy_demand']*100:.1f}% total patient demand)")
    print(f"  Days Exceeding 90% Beds    : {result['days_above_90pct']} of 60 days ({result['pct_days_above_90']:.1f}%)")
    print(f"  Total Bed-Deficit Days     : {result['surge_shortfall_days']} days with unmet admission demand")

def demo_predict_ward():
    print_separator("Demo 5: Ward-Level Bed-Day Aggregation (CLT Error Cancellation)")
    print("Scenario: Comparing aggregated bed-day prediction error for N=1 vs N=20 cohort.")
    
    pats_1 = [{"Department": "gynecology", "Severity of Illness": "Moderate"}]
    res_1 = predict_ward(pats_1, capacity=20)
    
    pats_20 = [{"Department": "gynecology", "Severity of Illness": "Moderate"} for _ in range(20)]
    res_20 = predict_ward(pats_20, capacity=20)
    
    print("\n[CLT Error Reduction Benchmark]")
    print(f"  Single Patient (N=1)   : Expected Days = {res_1['total_expected_bed_days']:.1f}d | Error = {res_1['typical_error_pct']:.2f}% | Reliable = {res_1['reliable']}")
    print(f"  Ward Cohort (N=20)     : Expected Days = {res_20['total_expected_bed_days']:.1f}d | Error = {res_20['typical_error_pct']:.2f}% | Reliable = {res_20['reliable']}")
    print(f"  Relative Improvement  : Error drops by {((res_1['typical_error_pct'] - res_20['typical_error_pct']) / res_1['typical_error_pct'])*100:.1f}% through soft aggregation")
    print(f"  Clinical Accounting    : {res_20['note']}")

def demo_get_evidence():
    print_separator("Demo 6: Self-Documenting Evidence Engine")
    ev = get_evidence()
    print(f"Status: {ev['status'].upper()} | Evidence Level: {ev['evidence_level'].upper()}")
    print("\n[Headline Model Accuracy vs Baselines]")
    stay_11 = ev['stay_accuracy_vs_baseline']['11_band']
    stay_3 = ev['stay_accuracy_vs_baseline']['3_class']
    print(f"  11-Band Stay Accuracy  : {stay_11['accuracy']*100:.2f}% vs {stay_11['baseline_accuracy']*100:.2f}% baseline (+{stay_11['delta_pp']:.2f} pp)")
    print(f"  3-Class Accuracy       : {stay_3['accuracy']*100:.2f}% vs {stay_3['baseline_accuracy']*100:.2f}% baseline (+{stay_3['delta_pp']:.2f} pp)")
    alert = ev['long_stay_alert']
    print(f"  Long-Stay Alert Recall : {alert['recall']*100:.2f}% at threshold {alert['threshold']} (ROC-AUC {alert['roc_auc']:.4f})")
    sp = ev['sparcs_result']
    print(f"  SPARCS Generalization  : {sp['ann_accuracy']*100:.2f}% vs {sp['baseline_accuracy']*100:.2f}% baseline ({sp['verdict']})")
    hmm = ev['hmm_vs_85_rule']
    print(f"  HMM Early-Warning      : {hmm['verdict']}")
    fc = ev['forecast_vs_persistence']
    print(f"  7-Day Forecast Verdict : {fc['verdict']}")
    print(f"\n[Documented Limitations ({len(ev['limitations'])} Core Disclaimers)]")
    for i, lim in enumerate(ev['limitations'], 1):
        print(f"  {i}. {lim[:80]}...")

def main():
    print("\n" + "#" * 78)
    print("  WARDCAST ML CORE: OPERATIONAL DEMONSTRATION & VERIFICATION")
    print("  University Course: PCCAIML 502 (Introduction to Machine Learning)")
    print("#" * 78)
    
    demo_predict_stay()
    demo_forecast_7day()
    demo_surge_state()
    demo_flu_scenario()
    demo_predict_ward()
    demo_get_evidence()
    
    print("\n" + "=" * 78)
    print("  DEMONSTRATION COMPLETE: All 6 ML prediction functions executed cleanly.")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
