"""
app/api.py - WardCast FastAPI Web Service
Provides production endpoints for inpatient length of stay, 7-day bed census forecasting,
HMM operational surge regime decoding, ward-level summing, and self-documenting evidence.

All ML models are loaded once at startup.
All inputs are strictly validated with human-readable error messages and zero stack traces.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request, Query, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.predict import (
    predict_stay,
    forecast_7day,
    surge_state,
    run_scenario,
    predict_ward,
    get_evidence,
    get_loaded_models
)

# Startup & Shutdown Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload all ML model artifacts once at startup
    print("Preloading WardCast ML models into memory...")
    try:
        models = get_loaded_models()
        print(f"ML models loaded successfully: {list(models.keys())}")
    except Exception as e:
        print(f"Warning during model preload: {e}")
    yield
    print("Shutting down WardCast service...")

app = FastAPI(
    title="WardCast Clinical Intelligence API",
    description="Operational Decision Support and Bed Management API for Hospital Operations",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Validation Error Handler (User-Friendly JSON, No Raw Stack Traces)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join([str(loc) for loc in err.get("loc", [])])
        msg = err.get("msg", "Invalid parameter")
        errors.append(f"{field_loc}: {msg}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": "Input validation failed. " + "; ".join(errors),
            "details": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": "HTTPException",
            "message": str(exc.detail)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": f"An operational error occurred processing the request: {str(exc)}"
        }
    )

# Pydantic Request Models
class PatientCharacteristics(BaseModel):
    Department: Optional[str] = Field(default="radiotherapy")
    Severity_of_Illness: Optional[str] = Field(default="Extreme", alias="Severity of Illness")
    Age: Optional[str] = Field(default="51-60")
    Visitors_with_Patient: Optional[int] = Field(default=6, ge=0, le=100, alias="Visitors with Patient")
    Admission_Deposit: Optional[float] = Field(default=6200.0, ge=0.0, alias="Admission_Deposit")
    Type_of_Admission: Optional[str] = Field(default="Emergency", alias="Type of Admission")
    Bed_Grade: Optional[float] = Field(default=3.0, ge=1.0, le=4.0, alias="Bed Grade")

    model_config = ConfigDict(populate_by_name=True, extra="allow")

class ForecastRequest(BaseModel):
    current_occupancy: Optional[float] = Field(default=0.78, ge=0.0, le=2.0)
    beds: Optional[int] = Field(default=120, ge=1, le=5000)
    recent_history: Optional[List[float]] = Field(default=None)
    patients: Optional[List[Dict[str, Any]]] = Field(default=None)

class SurgeStateRequest(BaseModel):
    occupancy_history: List[float] = Field(
        ...,
        min_length=1,
        description="Chronological sequence of occupancy rates (e.g. [0.74, 0.78, 0.82, 0.86, 0.89, 0.92])"
    )

    @field_validator("occupancy_history")
    def validate_rates(cls, v):
        for rate in v:
            if rate < 0.0 or rate > 2.0:
                raise ValueError(f"Occupancy rate {rate} outside realistic physical range [0.0, 2.0]")
        return v

class ScenarioRequest(BaseModel):
    name: Optional[str] = Field(default="flu_season_rush")
    capacity: Optional[int] = Field(default=120, ge=10, le=5000)
    multiplier: Optional[float] = Field(default=1.35, ge=1.0, le=3.0)

class WardCohortRequest(BaseModel):
    patients: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of admitted patient characteristic dictionaries"
    )
    capacity: Optional[int] = Field(default=None, ge=1)

# API Endpoints
@app.post("/api/predict_stay")
async def api_predict_stay(patient: Dict[str, Any]):
    """Predict length of stay distribution and long-stay alert for a single patient."""
    try:
        result = predict_stay(patient)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to predict stay: {str(e)}")

@app.post("/api/forecast_7day")
async def api_forecast_7day(req: ForecastRequest):
    """Generate 7-day rolling inpatient census forecast with empirical coverage intervals."""
    try:
        result = forecast_7day(
            patients=req.patients,
            current_occupancy=req.current_occupancy,
            beds=req.beds,
            recent_history=req.recent_history
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate forecast: {str(e)}")

@app.get("/api/forecast_7day")
async def api_forecast_7day_get(
    current_occupancy: float = Query(0.78, ge=0.0, le=2.0),
    beds: int = Query(120, ge=1, le=5000)
):
    """GET convenience endpoint for 7-day census forecast."""
    try:
        result = forecast_7day(current_occupancy=current_occupancy, beds=beds)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate forecast: {str(e)}")

@app.post("/api/surge_state")
async def api_surge_state(req: SurgeStateRequest):
    """Decode HMM operational regime and evaluate rule-based surge warning level."""
    try:
        result = surge_state(req.occupancy_history)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode surge state: {str(e)}")

@app.get("/api/surge_state")
async def api_surge_state_get(
    rates: Optional[str] = Query(None, description="Comma-separated rates e.g. 0.74,0.78,0.82,0.86,0.89,0.92")
):
    """GET convenience endpoint for surge state decoding."""
    try:
        if rates:
            occ_list = [float(x.strip()) for x in rates.split(",") if x.strip()]
        else:
            occ_list = [0.74, 0.78, 0.82, 0.86, 0.89, 0.92]
        result = surge_state(occ_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid occupancy sequence: {str(e)}")

@app.post("/api/run_scenario")
async def api_run_scenario(req: ScenarioRequest):
    """Simulate operational stress scenario (e.g. winter flu rush)."""
    try:
        result = run_scenario(name=req.name, capacity=req.capacity, multiplier=req.multiplier)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to run scenario: {str(e)}")

@app.get("/api/run_scenario")
async def api_run_scenario_get(
    name: str = Query("flu_season_rush"),
    capacity: int = Query(120, ge=10, le=5000),
    multiplier: float = Query(1.35, ge=1.0, le=3.0)
):
    """GET convenience endpoint for running stress scenarios."""
    try:
        result = run_scenario(name=name, capacity=capacity, multiplier=multiplier)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to run scenario: {str(e)}")

@app.post("/api/predict_ward")
async def api_predict_ward(req: WardCohortRequest):
    """Compute aggregated bed-day requirements for a ward cohort (CLT error cancellation)."""
    try:
        result = predict_ward(patients=req.patients, capacity=req.capacity)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to predict ward demand: {str(e)}")

@app.get("/api/get_evidence")
async def api_get_evidence():
    """Retrieve verified headline validation metrics, baselines, and clinical limitations."""
    try:
        result = get_evidence()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evidence: {str(e)}")

@app.get("/api/health")
async def api_health():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "WardCast Clinical Intelligence API",
        "evidence_contract_version": "1.0.0"
    }

# Static Files & Frontend Mounts
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index(tab: str = Query("today")):
    """Serve master clinical desktop interface (1440x900)."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        return JSONResponse(status_code=404, content={"message": "Frontend static file index.html not found"})
    
    if tab not in ["new-patient", "what-if", "how-accurate"]:
        return FileResponse(index_file)
    
    with open(index_file, "r", encoding="utf-8") as f:
        html = f.read()
    
    # Switch active tab pane
    html = html.replace('id="tab-today" class="tab-pane active', 'id="tab-today" class="tab-pane')
    html = html.replace(f'id="tab-{tab}" class="tab-pane', f'id="tab-{tab}" class="tab-pane active')
    
    # Switch active nav styling
    active_cls = 'h-full flex items-center font-semibold text-[#0B5CAD] border-b-[3px] border-[#0B5CAD] px-1 transition-colors'
    inactive_cls = 'h-full flex items-center font-normal text-[#475569] hover:text-[#0F172A] px-1 transition-colors'
    
    html = html.replace(f'class="{active_cls}" data-tab-target="today"', f'class="{inactive_cls}" data-tab-target="today"')
    html = html.replace(f'class="{inactive_cls}" data-tab-target="{tab}"', f'class="{active_cls}" data-tab-target="{tab}"')
    
    return HTMLResponse(content=html)

@app.get("/board")
async def serve_board():
    """Serve wall TV monitor display (1920x1080)."""
    board_file = os.path.join(STATIC_DIR, "board.html")
    if os.path.exists(board_file):
        return FileResponse(board_file)
    return JSONResponse(status_code=404, content={"message": "Frontend static file board.html not found"})
