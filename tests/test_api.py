"""
tests/test_api.py - Unit and Integration Tests for WardCast FastAPI Service
Verifies:
- One test per API endpoint: predict_stay, forecast_7day, surge_state, run_scenario, predict_ward, get_evidence
- Input validation and clear error responses (HTTP 422/400) without raw stack traces
- Static page endpoints for / and /board
"""

import unittest
from fastapi.testclient import TestClient
from app.api import app

class TestWardCastAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        """Verify service health endpoint."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_endpoint_predict_stay(self):
        """Verify POST /api/predict_stay with valid patient characteristics."""
        patient = {
            "Department": "radiotherapy",
            "Severity of Illness": "Extreme",
            "Age": "51-60",
            "Visitors with Patient": 6,
            "Admission_Deposit": 6200.0,
            "Type of Admission": "Emergency",
            "Bed Grade": 3.0
        }
        res = self.client.post("/api/predict_stay", json=patient)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["evidence_level"], "validated")
        self.assertIn("predicted_band", data)
        self.assertIn("class_3_probabilities", data)
        self.assertIn("top_factors", data)
        self.assertIn("associated with", data["top_factors"][0]["impact"])

    def test_endpoint_forecast_7day_post_and_get(self):
        """Verify POST and GET /api/forecast_7day endpoints."""
        # 1. POST
        payload = {
            "current_occupancy": 0.78,
            "beds": 120,
            "recent_history": [0.70, 0.72, 0.75, 0.78]
        }
        res_post = self.client.post("/api/forecast_7day", json=payload)
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertEqual(data_post["status"], "success")
        self.assertEqual(data_post["evidence_level"], "experimental")
        self.assertTrue(data_post["is_simulated"])
        self.assertEqual(data_post["average_stay_days"], 32.3)
        self.assertEqual(data_post["empirical_coverage"], 0.733)
        self.assertEqual(len(data_post["days"]), 7)
        self.assertIn("range_low", data_post["days"][0])
        self.assertIn("range_high", data_post["days"][0])

        # 2. GET
        res_get = self.client.get("/api/forecast_7day?current_occupancy=0.75&beds=120")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["status"], "success")
        self.assertEqual(data_get["forecast_basis"], "persistence")

    def test_endpoint_surge_state_post_and_get(self):
        """Verify POST and GET /api/surge_state endpoints."""
        # 1. POST
        payload = {"occupancy_history": [0.74, 0.78, 0.82, 0.86, 0.89, 0.92]}
        res_post = self.client.post("/api/surge_state", json=payload)
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertEqual(data_post["status"], "success")
        self.assertEqual(data_post["rule_warning_level"], "CRITICAL SURGE")
        self.assertIn("hmm_warning_level", data_post)
        self.assertEqual(data_post["warning_level"], "CRITICAL SURGE")
        self.assertIn("CRITICAL SURGE", data_post["warning_reason"])
        self.assertAlmostEqual(sum(data_post["current_state_posteriors"].values()), 1.0, places=4)

        # 2. GET
        res_get = self.client.get("/api/surge_state?rates=0.65,0.70,0.72")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertEqual(data_get["rule_warning_level"], "NORMAL")

    def test_endpoint_run_scenario_post_and_get(self):
        """Verify POST and GET /api/run_scenario stress test endpoints."""
        res = self.client.get("/api/run_scenario?multiplier=1.35&capacity=120")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["evidence_level"], "experimental")
        self.assertTrue(data["is_simulated"])
        self.assertEqual(data["average_stay_days"], 32.3)
        self.assertLessEqual(data["peak_occupancy_rate"], 1.0)
        self.assertGreaterEqual(data["peak_unmet_demand_beds"], 0)

    def test_endpoint_predict_ward(self):
        """Verify POST /api/predict_ward endpoint."""
        payload = {
            "patients": [
                {"Department": "gynecology", "Severity of Illness": "Moderate"}
                for _ in range(20)
            ],
            "capacity": 20
        }
        res = self.client.post("/api/predict_ward", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["evidence_level"], "validated")
        self.assertEqual(data["patient_count"], 20)
        self.assertTrue(data["reliable"])
        self.assertEqual(data["typical_error_pct"], 8.51)
        self.assertNotIn("projected_bed_turnover_days", data)
        self.assertIn("total_expected_bed_days represents the full length of stay", data["note"])

    def test_endpoint_get_evidence(self):
        """Verify GET /api/get_evidence self-documenting endpoint."""
        res = self.client.get("/api/get_evidence")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["evidence_level"], "validated")
        self.assertIn("stay_accuracy_vs_baseline", data)
        self.assertIn("long_stay_alert", data)
        self.assertIn("ward_error_by_n", data)
        self.assertIn("hmm_vs_85_rule", data)
        self.assertIn("forecast_vs_persistence", data)
        self.assertIn("sparcs_result", data)
        self.assertEqual(len(data["limitations"]), 5)

    def test_static_pages_served(self):
        """Verify / and /board static HTML routes serve valid content."""
        # 1. Desktop Index
        res_idx = self.client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        self.assertIn("text/html", res_idx.headers.get("content-type", ""))
        self.assertIn("WardCast", res_idx.text)
        self.assertIn("Simulated demo hospital (long-stay, average stay 32 days)", res_idx.text)

        # 2. Wall Board
        res_board = self.client.get("/board")
        self.assertEqual(res_board.status_code, 200)
        self.assertIn("text/html", res_board.headers.get("content-type", ""))
        self.assertIn("Ward Status Monitor", res_board.text)
        self.assertIn("Simulated demo hospital (long-stay, average stay 32 days)", res_board.text)

    def test_bad_input_handling_and_no_stack_traces(self):
        """Verify bad inputs return HTTP 422 with clear error messages and zero stack traces."""
        # 1. Empty occupancy history list
        res_empty = self.client.post("/api/surge_state", json={"occupancy_history": []})
        self.assertEqual(res_empty.status_code, 422)
        data_empty = res_empty.json()
        self.assertEqual(data_empty["status"], "error")
        self.assertEqual(data_empty["error_type"], "ValidationError")
        self.assertIn("Input validation failed", data_empty["message"])
        self.assertNotIn("Traceback", res_empty.text)

        # 2. Negative beds in forecast
        res_neg = self.client.post("/api/forecast_7day", json={"beds": -10})
        self.assertEqual(res_neg.status_code, 422)
        data_neg = res_neg.json()
        self.assertEqual(data_neg["status"], "error")
        self.assertNotIn("Traceback", res_neg.text)

        # 3. Empty ward patient list
        res_ward_empty = self.client.post("/api/predict_ward", json={"patients": []})
        self.assertEqual(res_ward_empty.status_code, 422)
        data_ward = res_ward_empty.json()
        self.assertEqual(data_ward["status"], "error")
        self.assertNotIn("Traceback", res_ward_empty.text)

if __name__ == "__main__":
    unittest.main()
