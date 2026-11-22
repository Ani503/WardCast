"""
src/generate_hmm_viterbi_plot.py
Step 2: Generate Viterbi-decoded state path figure over Rhode Island daily occupancy series.
Document Baum-Welch (EM with forward-backward) and Viterbi algorithms.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.fix_step_d import OrderedGaussianHMM4, prepare_features, STATE_NAMES_4

def main():
    print("Loading saved HMM model artifact...")
    model_path = "models/hmm_surge.joblib"
    hhs_path = "Data/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv"
    
    # Load model
    m_dict = joblib.load(model_path)
    raw_model = m_dict["raw_model"]
    state_perm = m_dict["state_perm"]
    inv_perm = m_dict["inv_perm"]
    state_names = m_dict.get("state_names", STATE_NAMES_4)
    
    # Load Rhode Island series
    print(f"Loading Rhode Island time series from {hhs_path}...")
    df_hhs = pd.read_csv(hhs_path, usecols=["state", "date", "inpatient_beds", "inpatient_beds_used"], low_memory=False)
    df_hhs["date"] = pd.to_datetime(df_hhs["date"], format="mixed")
    df_hhs = df_hhs[(df_hhs["inpatient_beds"] > 0) & df_hhs["inpatient_beds_used"].notnull()]
    df_hhs["occ"] = df_hhs["inpatient_beds_used"] / df_hhs["inpatient_beds"]
    
    df_ri = df_hhs[df_hhs["state"] == "RI"].sort_values("date").set_index("date")
    ri_daily = np.clip(df_ri["occ"].resample("D").mean().dropna().values, 0.2, 1.2)
    ri_dates = df_ri["occ"].resample("D").mean().dropna().index
    X_ri = prepare_features(ri_daily)
    
    # Viterbi decoding:
    # In hmmlearn, .predict() computes the most probable state sequence via the Viterbi algorithm.
    raw_states = raw_model.predict(X_ri)
    ordered_states = inv_perm[raw_states]
    
    print(f"Rhode Island series: {len(ri_daily)} days from {ri_dates[0].strftime('%Y-%m-%d')} to {ri_dates[-1].strftime('%Y-%m-%d')}.")
    unique_states, counts = np.unique(ordered_states, return_counts=True)
    for s, c in zip(unique_states, counts):
        print(f"  State {s} ({state_names[s]}): {c} days ({c/len(ri_daily)*100:.1f}%)")
        
    # Plot dual-panel Viterbi trajectory
    # Focus on a representative 180-day window showing normal, pre-surge, and critical surge transitions
    # (Days 300 to 500 contains acute winter surges in RI)
    eval_slice = slice(250, 450)
    slice_dates = ri_dates[eval_slice]
    slice_occ = ri_daily[eval_slice] * 100.0
    slice_states = ordered_states[eval_slice]
    n_pts = len(slice_occ)
    
    colors_4 = {0: "#a1d99b", 1: "#fed976", 2: "#fd8d3c", 3: "#e31a1c"}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # Upper panel: Occupancy time series with state background shading
    ax1.plot(slice_dates, slice_occ, color="#111111", lw=2.2, label="Actual Occupancy (%)", zorder=4)
    ax1.axhline(90, color="#b30000", linestyle="--", lw=1.8, label="90% Surge Threshold", zorder=5)
    ax1.axhline(85, color="#d95f0e", linestyle=":", lw=1.8, label="85% Alert Line", zorder=5)
    
    for i in range(n_pts - 1):
        ax1.axvspan(slice_dates[i], slice_dates[i+1], color=colors_4[slice_states[i]], alpha=0.30, lw=0)
        ax2.axvspan(slice_dates[i], slice_dates[i+1], color=colors_4[slice_states[i]], alpha=0.30, lw=0)
    # last point
    ax1.axvspan(slice_dates[-2], slice_dates[-1], color=colors_4[slice_states[-1]], alpha=0.30, lw=0)
    ax2.axvspan(slice_dates[-2], slice_dates[-1], color=colors_4[slice_states[-1]], alpha=0.30, lw=0)
    
    ax1.set_title("Rhode Island Hospital Capacity: Viterbi-Decoded HMM State Path & Bed Surge Detection\n"
                  "Parameter Estimation: Baum-Welch (EM with Forward-Backward) | State Decoding: Viterbi Algorithm",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Inpatient Occupancy Rate (%)", fontsize=11)
    ax1.set_ylim(60, 105)
    ax1.grid(True, alpha=0.3)
    
    p0 = mpatches.Patch(color=colors_4[0], alpha=0.6, label="State 0: Normal (Mean ~66.2%)")
    p1 = mpatches.Patch(color=colors_4[1], alpha=0.6, label="State 1: Moderate (Mean ~81.3%)")
    p2 = mpatches.Patch(color=colors_4[2], alpha=0.6, label="State 2: Pre-Surge (Mean ~86.7%)")
    p3 = mpatches.Patch(color=colors_4[3], alpha=0.6, label="State 3: Critical Surge (Mean ~90.0%)")
    ax1.legend(handles=[ax1.lines[0], ax1.lines[1], ax1.lines[2], p0, p1, p2, p3], loc="upper left", ncol=3, fontsize=9.5)
    
    # Lower panel: Discrete Viterbi MAP state path
    ax2.step(slice_dates, slice_states, where="mid", color="#08519c", lw=2.2, label="Viterbi MAP State Path")
    ax2.set_ylabel("Hidden State", fontsize=11)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["0 (Normal)", "1 (Mod)", "2 (Pre)", "3 (Surge)"], fontsize=9)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlabel("Timeline Date", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9.5)
    
    plt.tight_layout()
    fig_path = "reports/figures/hmm_viterbi_state_path.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Viterbi state path figure to {fig_path}")
    
    # Also update metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    if "step11_hmm_surge" not in metrics:
        metrics["step11_hmm_surge"] = {}
        
    metrics["step11_hmm_surge"]["algorithms"] = {
        "fitting_algorithm": "Baum-Welch algorithm (Expectation-Maximization using forward-backward probabilities)",
        "decoding_algorithm": "Viterbi algorithm (dynamic programming maximum a posteriori path estimation)",
        "viterbi_figure": "reports/figures/hmm_viterbi_state_path.png",
        "rhode_island_state_distribution": {
            "Normal (State 0)": int(counts[0]),
            "Moderate (State 1)": int(counts[1]),
            "Pre-Surge (State 2)": int(counts[2]),
            "Critical Surge (State 3)": int(counts[3]),
            "total_days": int(len(ri_daily))
        }
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path} with HMM algorithms specification.")

if __name__ == "__main__":
    main()
