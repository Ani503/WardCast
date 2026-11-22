"""
tests/test_predict.py
Unit tests verifying the clean prediction interface in src/predict.py.
Tests all 4 required operational API endpoints:
- predict_stay()
- forecast_7d()
- surge_state()
- run_scenario()
"""

import unittest
import numpy as np
from src.predict import (
    predict_stay,
    forecast_7day,
    forecast_7d,
    surge_state,
    flu_scenario,
    run_scenario,
    predict_ward,
    get_evidence,
    STAY_BANDS
)

class TestPredictInterface(unittest.TestCase):

    def test_predict_stay_full_features(self):
        patient = {
            "Department": "radiotherapy",
            "Severity of Illness": "Extreme",
            "Age": "51-60",
            "Visitors with Patient": 6,
            "Admission_Deposit": 6200.0,
            "Type of Admission": "Emergency",
            "Bed Grade": 3.0
        }
        res = predict_stay(patient)
        self.assertEqual(res["status"], "success")
        self.assertIn("predicted_band", res)
        self.assertIn(res["predicted_band"], STAY_BANDS)
        self.assertIsInstance(res["predicted_days_midpoint"], float)
        self.assertGreater(res["predicted_days_midpoint"], 0)
        
        # Verify 11 band probabilities
        self.assertIn("band_probabilities", res)
        self.assertEqual(len(res["band_probabilities"]), 11)
        prob_sum = sum(res["band_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
        # Verify clustering and factors
        self.assertIn("cluster_id", res)
        self.assertIn(res["cluster_id"], [0, 1, 2])
        self.assertIn("cluster_label", res)
        self.assertIsInstance(res["top_factors"], list)
        self.assertGreater(len(res["top_factors"]), 0)
        self.assertIn("associated with", res["top_factors"][0]["impact"])
        self.assertNotIn("Driver", res["top_factors"][0]["impact"])
        self.assertNotIn("clinical_rationale", res["top_factors"][0])

    def test_predict_stay_missing_features(self):
        # Empty dict should fallback to population defaults gracefully
        res = predict_stay({})
        self.assertEqual(res["status"], "success")
        self.assertIn(res["predicted_band"], STAY_BANDS)
        self.assertIn("cluster_id", res)
        self.assertEqual(len(res["band_probabilities"]), 11)

    def test_forecast_7d_standard(self):
        patients = [
            {"patient_id": f"P-{i}", "Department": "gynecology", "Severity of Illness": "Moderate", "days_spent": i % 15}
            for i in range(1, 85)
        ]
        res = forecast_7d(patients, capacity=120)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["current_census"], 84)
        self.assertEqual(res["capacity"], 120)
        self.assertIn(res["warning_level"], ["NORMAL (GREEN)", "WARNING (YELLOW)", "CRITICAL SURGE (RED)"])
        self.assertIn("days", res)
        self.assertEqual(len(res["days"]), 7)
        
        for day in res["days"]:
            self.assertIn("horizon_day", day)
            self.assertIn("median_census", day)
            self.assertIn("ci_low_80", day)
            self.assertIn("ci_high_80", day)
            self.assertIn("median_occupancy_rate", day)
            self.assertIn("shortfall_beds", day)
            self.assertIn("shortfall_probability", day)
            self.assertIn("surge_probability", day)
            # Verify ordering of CI
            self.assertLessEqual(day["ci_low_80"], day["median_census"])
            self.assertLessEqual(day["median_census"], day["ci_high_80"])
            self.assertGreaterEqual(day["shortfall_beds"], 0)

    def test_forecast_7d_empty_hospital(self):
        # Edge case: empty hospital
        res = forecast_7d([], capacity=100)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["current_census"], 0)
        self.assertEqual(len(res["days"]), 7)
        self.assertEqual(res["warning_level"], "NORMAL (GREEN)")

    def test_surge_state_normal(self):
        occ_hist = [0.65, 0.68, 0.70, 0.72]
        res = surge_state(occ_hist)
        self.assertEqual(res["status"], "success")
        self.assertIn("current_state_id", res)
        self.assertIn("current_state_name", res)
        self.assertIn(res["current_state_name"], ["Normal", "Moderate", "Pre-Surge", "Critical Surge"])
        self.assertIn("current_state_posteriors", res)
        self.assertIn("next_day_surge_probability", res)
        self.assertGreaterEqual(res["next_day_surge_probability"], 0.0)
        self.assertLessEqual(res["next_day_surge_probability"], 1.0)
        self.assertIn("recommended_action", res)
        self.assertEqual(res["rule_warning_level"], "NORMAL")
        self.assertIn(res["hmm_warning_level"], ["NORMAL", "SURGE ALERT (PRE-SURGE)", "CRITICAL SURGE"])
        self.assertEqual(res["warning_level"], res["rule_warning_level"])
        self.assertIsInstance(res["warning_reason"], str)
        self.assertIn("NORMAL", res["warning_reason"])

    def test_surge_state_surge_crisis(self):
        occ_hist = [0.88, 0.91, 0.95, 0.97]
        res = surge_state(occ_hist)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["next_day_surge_probability"], 0.50)
        self.assertIn("CRITICAL SURGE", res["recommended_action"])
        self.assertEqual(res["rule_warning_level"], "CRITICAL SURGE")
        self.assertEqual(res["warning_level"], "CRITICAL SURGE")
        self.assertIn("CRITICAL SURGE", res["warning_reason"])

    def test_surge_state_edge_cases(self):
        # Single element history
        res_single = surge_state([0.75])
        self.assertEqual(res_single["status"], "success")
        
        # Empty list history
        res_empty = surge_state([])
        self.assertEqual(res_empty["status"], "success")

    def test_forecast_7day_occupancy_input(self):
        # Tests forecast_7day(current_occupancy=0.75, beds=120)
        res = forecast_7day(current_occupancy=0.75, beds=120)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["capacity"], 120)
        self.assertEqual(res["forecast_basis"], "persistence")
        self.assertEqual(res["empirical_coverage"], 0.733)
        self.assertTrue(res["is_simulated"])
        self.assertEqual(res["average_stay_days"], 32.3)
        self.assertEqual(len(res["daily_occupancy"]), 7)
        self.assertEqual(len(res["surge_risk_per_day"]), 7)
        self.assertLessEqual(res["current_occupancy"], 1.0)
        self.assertIn("unmet_demand_beds", res)
        for d in res["days"]:
            self.assertIn("range_low", d)
            self.assertIn("range_high", d)
            self.assertLessEqual(d["range_low"], d["range_high"])
        for occ in res["daily_occupancy"]:
            self.assertGreaterEqual(occ, 0.0)
            self.assertLessEqual(occ, 1.0)
        for risk in res["surge_risk_per_day"]:
            self.assertGreaterEqual(risk, 0.0)
            self.assertLessEqual(risk, 1.0)
            
    def test_flu_scenario_stress(self):
        # Tests flu_scenario(beds=120, multiplier=1.35)
        res = flu_scenario(beds=120, multiplier=1.35)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["beds"], 120)
        self.assertEqual(res["multiplier"], 1.35)
        self.assertGreater(res["surge_occupancy"], res["baseline_occupancy"])
        self.assertGreater(res["occupancy_change_pp"], 0.0)
        self.assertGreaterEqual(res["days_above_90pct"], 0)
        self.assertLessEqual(res["days_above_90pct"], 60)

    def test_run_scenario(self):
        res = run_scenario("flu_season_rush", capacity=120)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_simulated"])
        self.assertEqual(res["average_stay_days"], 32.3)
        self.assertEqual(res["scenario_name"], "flu_season_rush")
        self.assertEqual(res["configured_capacity"], 120)
        self.assertGreater(res["mean_occupancy_rate"], 0.0)
        self.assertGreater(res["peak_occupancy_rate"], 0.0)
        self.assertIsInstance(res["surge_days_count"], int)
        self.assertIsInstance(res["shortfall_days_count"], int)
        self.assertIsInstance(res["trajectory"], list)
        self.assertGreater(len(res["trajectory"]), 0)

    def test_predict_ward(self):
        # 1. Single patient: N=1, reliable=False, typical_error_pct=56.54%
        pats_1 = [{"Department": "gynecology", "Severity of Illness": "Moderate"}]
        res_1 = predict_ward(pats_1, capacity=20)
        self.assertEqual(res_1["status"], "success")
        self.assertEqual(res_1["evidence_level"], "validated")
        self.assertEqual(res_1["patient_count"], 1)
        self.assertFalse(res_1["reliable"])
        self.assertEqual(res_1["typical_error_pct"], 56.54)
        self.assertNotIn("projected_bed_turnover_days", res_1)
        self.assertIn("total_expected_bed_days represents the full length of stay", res_1["note"])

        # 2. Cohort: N=20, reliable=True, typical_error_pct=8.51%
        pats_20 = [{"Department": "gynecology", "Severity of Illness": "Moderate"} for _ in range(20)]
        res_20 = predict_ward(pats_20, capacity=20)
        self.assertEqual(res_20["patient_count"], 20)
        self.assertTrue(res_20["reliable"])
        self.assertEqual(res_20["typical_error_pct"], 8.51)
        self.assertNotIn("projected_bed_turnover_days", res_20)

    def test_get_evidence(self):
        ev = get_evidence()
        self.assertEqual(ev["status"], "success")
        self.assertEqual(ev["evidence_level"], "validated")
        required_keys = [
            "stay_accuracy_vs_baseline",
            "long_stay_alert",
            "ward_error_by_n",
            "hmm_vs_85_rule",
            "forecast_vs_persistence",
            "sparcs_result",
            "limitations"
        ]
        for rk in required_keys:
            self.assertIn(rk, ev)
        self.assertEqual(len(ev["limitations"]), 5)

    def test_canonical_function_names(self):
        from src import predict
        expected_endpoints = {
            "predict_stay",
            "forecast_7day",
            "surge_state",
            "run_scenario",
            "predict_ward",
            "get_evidence"
        }
        self.assertEqual(set(predict.__all__), expected_endpoints)

    def test_predict_stay_probability_sum_precision(self):
        """Verify that predict_stay probabilities sum to 1.0 within 1e-6 precision."""
        test_cases = [
            {}, # default patient
            {"Department": "radiotherapy", "Severity of Illness": "Extreme", "Visitors with Patient": 6},
            {"Department": "gynecology", "Severity of Illness": "Minor", "Visitors with Patient": 1},
            {"Department": "surgery", "Severity of Illness": "Moderate", "Age": "71-80", "Admission_Deposit": 7500.0}
        ]
        for pat in test_cases:
            res = predict_stay(pat)
            prob_sum_11 = sum(res["band_probabilities"].values())
            self.assertAlmostEqual(prob_sum_11, 1.0, places=6,
                                   msg=f"11-band probabilities sum to {prob_sum_11}, expected 1.0 within 1e-6")
            
            prob_sum_3 = sum(res["class_3_probabilities"].values())
            self.assertAlmostEqual(prob_sum_3, 1.0, places=6,
                                   msg=f"3-class probabilities sum to {prob_sum_3}, expected 1.0 within 1e-6")

    def test_3class_label_definition_consistency(self):
        """Verify that 3-class labels strictly match Short (<=20d), Medium (21-50d), Long (>50d)."""
        from src.prep import BAND_TO_3CLASS, CLASS_3_NAMES, BAND_TO_MIDPOINT
        
        # Verify naming mapping
        self.assertEqual(CLASS_3_NAMES[0], "Short (0-20d)")
        self.assertEqual(CLASS_3_NAMES[1], "Medium (21-50d)")
        self.assertEqual(CLASS_3_NAMES[2], "Long (>50d)")
        
        # Verify boundary midpoints
        for band, class_idx in BAND_TO_3CLASS.items():
            mid = BAND_TO_MIDPOINT[band]
            if class_idx == 0:
                self.assertLessEqual(mid, 20.0, f"Band {band} in class 0 must have midpoint <= 20.0")
            elif class_idx == 1:
                self.assertTrue(21.0 <= mid <= 50.0, f"Band {band} in class 1 must have midpoint in [21, 50]")
            elif class_idx == 2:
                self.assertGreater(mid, 50.0, f"Band {band} in class 2 must have midpoint > 50.0")
                
        # Verify predict_stay returns valid documented labels
        res = predict_stay({"Department": "gynecology", "Severity of Illness": "Moderate"})
        self.assertIn(res["class_3_label"], list(CLASS_3_NAMES.values()))
        self.assertEqual(set(res["class_3_probabilities"].keys()), set(CLASS_3_NAMES.values()))

    def test_model_accuracy_report_metrics_json_consistency(self):
        """Verify that reports/model_accuracy_report.md exists and is consistent with metrics.json."""
        import json
        import os
        
        report_path = "reports/model_accuracy_report.md"
        metrics_path = "reports/metrics.json"
        
        self.assertTrue(os.path.exists(report_path), f"Missing {report_path}")
        self.assertTrue(os.path.exists(metrics_path), f"Missing {metrics_path}")
        
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
            
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
        # Verify critical dataset counts in report
        verified_counts = metrics_data["verified_dataset_counts"]
        kag_patients = str(verified_counts["kaggle_train_data"]["unique_patients"]) # "92017"
        kag_patients_fmt = f"{int(kag_patients):,}" # "92,017"
        kag_rows_fmt = f"{verified_counts['kaggle_train_data']['total_rows']:,}" # "318,438"
        sparcs_rows_fmt = f"{verified_counts['sparcs_2014']['total_rows_on_disk']:,}" # "2,367,550"
        hhs_rows_fmt = f"{verified_counts['hhs_state_timeseries']['valid_cleaned_rows']:,}" # "80,983"
        
        self.assertIn(kag_patients_fmt, report_text, f"Unique patients {kag_patients_fmt} missing from report")
        self.assertIn(kag_rows_fmt, report_text, f"Kaggle rows {kag_rows_fmt} missing from report")
        self.assertIn(sparcs_rows_fmt, report_text, f"SPARCS rows {sparcs_rows_fmt} missing from report")
        self.assertIn(hhs_rows_fmt, report_text, f"HHS valid rows {hhs_rows_fmt} missing from report")
        
        # Verify critical model accuracy numbers in report
        self.assertIn("42.49%", report_text, "ANN 11-band accuracy 42.49% missing from report")
        self.assertIn("27.48%", report_text, "Majority baseline accuracy 27.48% missing from report")
        self.assertIn("48.48%", report_text, "3-class baseline accuracy 48.48% missing from report")
        self.assertIn("62.63%", report_text, "3-class ANN accuracy 62.63% missing from report")
        self.assertIn("0.7123", report_text, "Isotonic ANN Brier score 0.7123 missing from report")
        self.assertIn("1.4505", report_text, "Majority Brier baseline 1.4505 missing from report")
        self.assertIn("77.51%", report_text, "Selected threshold 0.25 recall 77.51% missing from report")
        self.assertIn("89.09%", report_text, "SPARCS majority baseline 89.09% missing from report")
        self.assertIn("89.05%", report_text, "SPARCS ANN accuracy 89.05% missing from report")
        self.assertIn("76.65%", report_text, "Simulator mean occupancy 76.65% missing from report")
        self.assertIn("3.84%", report_text, "Ward summing N=100 MAPE 3.84% missing from report")

    def test_hmm_posteriors_sum_strictly_to_one(self):
        """Verify that HMM posteriors sum to 1.0 within 1e-4 across diverse occupancy histories and contain 4 unique states."""
        test_histories = [
            [],
            [0.72],
            [0.65, 0.68, 0.70, 0.72],
            [0.74, 0.78, 0.82, 0.86, 0.89, 0.92],
            [0.88, 0.91, 0.95, 0.97],
            [0.50, 0.55],
            [0.98, 0.99]
        ]
        for hist in test_histories:
            res = surge_state(hist)
            self.assertEqual(res["status"], "success")
            self.assertIn("current_state_posteriors", res)
            posts = res["current_state_posteriors"]
            self.assertEqual(len(posts), 4)
            self.assertEqual(set(posts.keys()), {"Normal", "Moderate", "Pre-Surge", "Critical Surge"})
            self.assertAlmostEqual(sum(posts.values()), 1.0, places=4)
            self.assertIn(res["current_state_name"], ["Normal", "Moderate", "Pre-Surge", "Critical Surge"])
            self.assertLessEqual(res["latest_occupancy_rate"], 1.0)
            self.assertGreaterEqual(res["unmet_demand_beds"], 0)

    def test_predict_occupancy_capping_and_unmet_demand(self):
        """Verify that predict.py endpoints never return physical bed occupancy > 1.0 (100%) and report unmet demand."""
        # 1. forecast_7day with overload occupancy
        fc_res = forecast_7day(current_occupancy=1.15, beds=120)
        self.assertLessEqual(fc_res["current_occupancy"], 1.0)
        self.assertGreaterEqual(fc_res["current_unmet_demand_beds"], 1)
        for occ in fc_res["daily_occupancy"]:
            self.assertLessEqual(occ, 1.0)
        for unmet in fc_res["unmet_demand_beds"]:
            self.assertGreaterEqual(unmet, 0)
            
        # 2. surge_state with overload history
        surge_res = surge_state([0.98, 1.15])
        self.assertLessEqual(surge_res["latest_occupancy_rate"], 1.0)
        self.assertGreaterEqual(surge_res["unmet_demand_beds"], 1)
        
        # 3. flu_scenario stress test
        flu_res = flu_scenario(beds=120, multiplier=1.50)
        self.assertLessEqual(flu_res["peak_occupancy"], 1.0)
        self.assertLessEqual(flu_res["surge_occupancy"], 1.0)
        self.assertLessEqual(flu_res["baseline_occupancy"], 1.0)
        self.assertGreaterEqual(flu_res["unmet_demand_beds"], 0)
        
        # 4. run_scenario simulation
        scen_res = run_scenario("flu_season_rush", capacity=120)
        self.assertLessEqual(scen_res["peak_occupancy_rate"], 1.0)
        self.assertLessEqual(scen_res["mean_occupancy_rate"], 1.0)
        self.assertGreaterEqual(scen_res["unmet_demand_beds"], 0)
        for t in scen_res["trajectory"]:
            self.assertLessEqual(t["occupancy_rate"], 1.0)
            self.assertGreaterEqual(t["unmet_demand_beds"], 0)

    def test_evidence_level_field_presence(self):
        """Verify that all predict functions tag their evidence_level as 'validated' or 'experimental'."""
        # predict_stay: validated
        stay_res = predict_stay({})
        self.assertEqual(stay_res.get("evidence_level"), "validated")
        
        # predict_ward: validated
        ward_res = predict_ward([{"Department": "gynecology"}])
        self.assertEqual(ward_res.get("evidence_level"), "validated")
        
        # forecast_7day / forecast_7d: experimental
        fc_res = forecast_7day(current_occupancy=0.75, beds=120)
        self.assertEqual(fc_res.get("evidence_level"), "experimental")
        
        # surge_state: experimental (HMM) and validated (warning rule)
        surge_res = surge_state([0.75, 0.78])
        self.assertEqual(surge_res.get("evidence_level"), "experimental")
        self.assertEqual(surge_res.get("warning_rule_evidence_level"), "validated")
        
        # flu_scenario: experimental
        flu_res = flu_scenario(beds=120, multiplier=1.35)
        self.assertEqual(flu_res.get("evidence_level"), "experimental")
        
        # run_scenario: experimental
        scen_res = run_scenario("flu_season_rush", capacity=120)
        self.assertEqual(scen_res.get("evidence_level"), "experimental")

        # get_evidence: validated
        ev_res = get_evidence()
        self.assertEqual(ev_res.get("evidence_level"), "validated")

    def test_model_selection_k_values_and_conclusions(self):
        """Verify that model selection k values and conclusions are consistent across metrics.json and reports."""
        import json
        with open("reports/metrics.json") as f:
            m = json.load(f)
            
        # 1. HMM k=4
        self.assertEqual(m["step11_hmm_surge"]["model_selection"]["chosen_k"], 4)
        with open("reports/hmm_surge_report.md") as f:
            hmm_text = f.read()
        self.assertIn("k=4", hmm_text)
        self.assertNotIn("k=3 selected", hmm_text.lower())
        self.assertNotIn("zero days of notice", hmm_text.lower())
        self.assertIn("no better than the 85% rule", hmm_text.lower())
        
        # 2. SPARCS EM k=6
        self.assertEqual(m["step6_sparcs_censored_em"]["selected_k"], 6)
        with open("reports/sparcs_validation.md") as f:
            sparcs_text = f.read()
        self.assertNotIn("generalize robustly", sparcs_text.lower())
        self.assertIn("majority-baseline level", sparcs_text.lower())
        
        # 3. Simulator baseline surge days
        self.assertEqual(m["simulator"]["surge_frequency_pct"], 3.93)
        with open("reports/simulator_validation.md") as f:
            sim_text = f.read()
        self.assertIn("3.93%", sim_text)

    def test_strict_number_extractor_reports_consistency(self):
        """
        Strict number test: extract EVERY number from ALL 12 markdown files in reports/*.md
        and fail if it is not present in metrics.json or the CSV tables.
        """
        import re
        import json
        import os
        import pandas as pd
        import numpy as np

        def collect_pool():
            pool_floats = set()

            def add_val(v):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    fv = float(v)
                    for val_variant in [fv, abs(fv), -abs(fv)]:
                        pool_floats.add(val_variant)
                        if abs(val_variant) <= 2.0:
                            pool_floats.add(round(val_variant * 100, 2))
                            pool_floats.add(round(val_variant * 100, 1))
                        pool_floats.add(round(val_variant, 4))
                        pool_floats.add(round(val_variant, 3))
                        pool_floats.add(round(val_variant, 2))
                        pool_floats.add(round(val_variant, 1))
                        pool_floats.add(round(val_variant, 0))
                elif isinstance(v, str):
                    clean = v.replace("$", "").replace("%", "").replace(",", "").replace("+", "").strip()
                    sub_nums = re.findall(r"[\+\-]?\d+(?:\.\d+)?", clean)
                    for sn in sub_nums:
                        try:
                            add_val(float(sn))
                        except ValueError:
                            pass

            def recurse(o):
                if isinstance(o, dict):
                    for k, val in o.items():
                        recurse(val)
                elif isinstance(o, list):
                    for item in o:
                        recurse(item)
                else:
                    add_val(o)

            with open("reports/metrics.json") as f:
                m = json.load(f)
            recurse(m)

            for p in ["reports/stay_model_comparison.csv", "reports/forecast_evaluation.csv",
                      "reports/forecast_real_hhs_evaluation.csv", "reports/ward_summing_evaluation.csv",
                      "reports/association_rules.csv"]:
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    for col in df.columns:
                        for val in df[col]:
                            add_val(val)

            # Allowed structural numbers (course code, dates, dimensions, section numbers)
            structural = [
                502, 2014, 2020, 2022, 2023, 2024, 20261121, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 23, 24,
                1.1, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 8.1, 8.2, 9.1, 9.2,
                64, 32, 116, 120, 109, 1095, 60, 30, 480, 5, 3, 11,
                0.15, 0.20, 0.25, 0.30, 0.333, 0.35, 0.40, 0.50, 0.85, 0.90, 0.92,
                1, 16, 20, 50, 100, 101, 103, 108, 110, 119, 119.5, 120, 136, 150,
                14, 10, 4.75, 0.01, 0.04, 0.20
            ]

            for s in structural:
                add_val(s)

            return pool_floats

        pool = collect_pool()

        report_files = [
            "reports/00_data_audit.md",
            "reports/FINAL_SUMMARY.md",
            "reports/UI_CONTRACT.md",
            "reports/association_rules.md",
            "reports/cluster_profiles.md",
            "reports/feature_dictionary.md",
            "reports/forecast_report.md",
            "reports/hhs_calibration.md",
            "reports/hmm_surge_report.md",
            "reports/model_accuracy_report.md",
            "reports/simulator_validation.md",
            "reports/sparcs_validation.md"
        ]

        for rf in report_files:
            self.assertTrue(os.path.exists(rf), f"Report file {rf} does not exist")
            with open(rf, "r", encoding="utf-8") as f:
                text = f.read()
            # Mask ISO dates so date hyphens are not parsed as negative numbers
            text_clean = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
            tokens = re.findall(r"[\+\-]?(?:\$)?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%|x)?\b|\b\d+\.\d+\b", text_clean)
            missing = []
            for raw in tokens:
                val_str = raw.replace("$", "").replace("%", "").replace("x", "").replace(",", "")
                try:
                    val = float(val_str)
                    matched = any(abs(val - pf) < 1e-3 or abs(val*100 - pf) < 1e-2 or abs(val - pf*100) < 1e-2 for pf in pool)
                    if not matched:
                        missing.append((raw, val))
                except ValueError:
                    pass
            self.assertEqual(len(missing), 0, f"Numbers in {rf} not found in metrics.json/CSVs: {set(missing)}")

if __name__ == "__main__":
    unittest.main()



