# WardCast Operational Application: Run & Deployment Guide

This document details installation, execution commands, application URLs, and test suites for the WardCast working application.

---

## 1. Environment & Installation

The application runs on Python 3.11 within the project's virtual environment.

```bash
# Activate existing virtual environment
source .venv/bin/activate

# Ensure core and web dependencies are installed
pip install -r requirements.txt
pip install fastapi uvicorn httpx
```

---

## 2. Running the Application Server

Start the production FastAPI backend with Uvicorn:

```bash
# Standard startup command (from project root)
PYTHONPATH=. .venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000

# For development with auto-reload:
PYTHONPATH=. .venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Web Application URLs

| Interface | URL | Display Target | Description |
|:---|:---|:---|:---|
| **Main Clinical App** | [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | 1440x900 Desktop | 4-tab clinical decision support console (Today, New Patient, What If, How Accurate) |
| **Today's Status Tab** | [http://127.0.0.1:8000/#today](http://127.0.0.1:8000/#today) | 1440x900 Desktop | Inpatient census, surge rule banner, 7-day forecast chart, HMM regime callout |
| **New Patient Tab** | [http://127.0.0.1:8000/#new-patient](http://127.0.0.1:8000/#new-patient) | 1440x900 Desktop | Admission triage, 3-class length of stay probabilities, long-stay alert engine |
| **What If Stress Tab** | [http://127.0.0.1:8000/#what-if](http://127.0.0.1:8000/#what-if) | 1440x900 Desktop | 60-day epidemic rush simulator with surge slider (+0% to +60%) and overflow tracking |
| **How Accurate Tab** | [http://127.0.0.1:8000/#how-accurate](http://127.0.0.1:8000/#how-accurate) | 1440x900 Desktop | Self-documenting evidence scorecard, validation baselines, and clinical limitations |
| **Ward Wall Display** | [http://127.0.0.1:8000/board](http://127.0.0.1:8000/board) | 1920x1080 Wall TV | Read-only high-contrast 16:9 monitor for ward coordination (no patient details) |
| **API Swagger UI** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Browser | Interactive OpenAPI specification and schema explorer |
| **API ReDoc** | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | Browser | Clean OpenAPI technical documentation |

---

## 4. API Endpoints Reference

All endpoints strictly follow the contract specified in `reports/UI_CONTRACT.md`:

- `POST /api/predict_stay`: Evaluates inpatient stay distribution and long-stay alert for an incoming patient.
- `POST /api/forecast_7day` & `GET /api/forecast_7day`: Returns 7-day rolling inpatient census projections and prediction intervals.
- `POST /api/surge_state` & `GET /api/surge_state`: Decodes HMM regime and computes rule-based surge warning levels.
- `POST /api/run_scenario` & `GET /api/run_scenario`: Executes 60-day epidemic stress test simulation.
- `POST /api/predict_ward`: Aggregates bed-day demand across patient cohorts (CLT error cancellation).
- `GET /api/get_evidence`: Returns self-documenting headline validation benchmarks and documented limitations.
- `GET /api/health`: Service health status.

---

## 5. Running the Test Suites

Execute all tests from the repository root:

```bash
# 1. Run all unit and integration tests (including ML and API suites)
PYTHONPATH=. .venv/bin/python -m pytest tests -q

# 2. Run API test suite specifically with verbose output
PYTHONPATH=. .venv/bin/python -m pytest tests/test_api.py -v

# 3. Run ML Core terminal demo
PYTHONPATH=. .venv/bin/python run_demo.py
```
