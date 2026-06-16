import os
import uuid
from uuid import UUID
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from sqlalchemy.orm import Session

# Import SQLAlchemy database setup and models
from src.database.connection import get_db, SessionLocal
from src.database import models as db_models

# Import ML inference and prognosis modules
from inference import run_inference
from prognosis_engine import get_prognosis

# Import Pydantic validation schemas
import schemas

# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Bone Fracture Detection & Prognosis Dashboard Backend",
    description="Production-grade FastAPI serving dual-head PyTorch classification, Grad-CAM overlays, and AO-guidelines prognosis rules.",
    version="2.0.0"
)

# CORS Middleware (allowing all origins for local dashboard development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
UPLOAD_DIR = Path("uploads")
HEATMAP_DIR = Path("heatmaps")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Mount uploads, heatmaps, and templates static directories to serve files in the browser
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# ------------------------------------------------------------------------------
# Frontend Page Routes
# ------------------------------------------------------------------------------

@app.get("/")
def serve_dashboard():
    """Serves the main diagnostics and prognosis registration dashboard page."""
    return FileResponse(Path("templates/index.html"))

@app.get("/history")
def serve_history():
    """Serves the patient search and scan history timeline page."""
    return FileResponse(Path("templates/history.html"))

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.post("/patients/", response_model=schemas.PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(patient_in: schemas.PatientCreate, db: Session = Depends(get_db)):
    """Registers a new patient clinical profile in the database."""
    new_patient = db_models.Patient(
        full_name=patient_in.full_name,
        age=patient_in.age,
        gender=patient_in.gender,
        comorbidities=patient_in.comorbidities
    )
    try:
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        return new_patient
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error registering patient: {e}"
        )

@app.post("/scan/upload", response_model=schemas.ScanUploadResponse)
async def upload_scan(
    request: Request,
    patient_id: str = Form(..., description="UUID of the patient"),
    file: UploadFile = File(..., description="X-ray scan image file"),
    db: Session = Depends(get_db)
):
    """
    Accepts an X-ray scan image, performs deep learning fracture detection & bone type 
    classification, generates a Grad-CAM heatmap overlay, calculates recovery guidelines, 
    saves all results to the database, and returns the integrated diagnostics response.
    """
    # 1. Fetch patient and verify existence
    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid patient_id format. Must be a valid UUID string."
        )

    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_uuid).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record with ID {patient_id} not found."
        )

    # 2. Save file to uploads/ with unique UUID name
    file_extension = Path(file.filename).suffix
    unique_filename = f"scan_{uuid.uuid4()}{file_extension}"
    saved_file_path = UPLOAD_DIR / unique_filename

    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write image file to uploads storage: {e}"
        )

    # 3. Create scan record in database (flushed to obtain scan.id)
    rel_file_path = f"uploads/{unique_filename}"
    new_scan = db_models.XrayScan(
        patient_id=patient.id,
        original_file_path=rel_file_path,
        bone_affected="temporary",  # Will update with prediction results
        image_quality_flag="Good",
        dataset_source=db_models.DatasetSource.uploaded
    )
    db.add(new_scan)
    db.flush()

    # 4. Run PyTorch model inference (returns InferenceResult)
    try:
        inference_res = run_inference(str(saved_file_path))
    except Exception as e:
        db.rollback()
        if saved_file_path.exists():
            os.remove(saved_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML Model classification execution failed: {e}"
        )

    # Update bone affected with predicted bone type
    new_scan.bone_affected = inference_res.bone_affected
    db.flush()

    # Resolve web heatmap relative path
    rel_heatmap_path = ""
    if inference_res.heatmap_path:
        rel_heatmap_path = f"heatmaps/{Path(inference_res.heatmap_path).name}"

    # 5. Log prediction in database
    db_severity = None
    if inference_res.fracture_detected:
        try:
            db_severity = db_models.Severity(inference_res.severity)
        except ValueError:
            pass

    new_prediction = db_models.FracturePrediction(
        scan_id=new_scan.id,
        fracture_detected=inference_res.fracture_detected,
        severity=db_severity,
        confidence_score=inference_res.severity_confidence,
        heatmap_path=rel_heatmap_path if rel_heatmap_path else None,
        model_version="v2.0.0"
    )
    db.add(new_prediction)
    db.flush()

    # 6. Execute AO rules-based prognosis calculations
    try:
        # If no fracture is detected, return baseline normal indicators
        if not inference_res.fracture_detected:
            rest_weeks_min = 0
            rest_weeks_max = 0
            cast_type = "None"
            plaster_required = False
            weight_bearing_status = "Full weight-bearing"
            referral_flag = "conservative"
        else:
            prog_res = get_prognosis(
                bone=inference_res.bone_affected,
                severity=inference_res.severity,
                age=patient.age,
                comorbidities=patient.comorbidities
            )
            rest_weeks_min = prog_res.rest_weeks_min
            rest_weeks_max = prog_res.rest_weeks_max
            cast_type = prog_res.cast_type
            plaster_required = prog_res.plaster_required
            weight_bearing_status = prog_res.weight_bearing_status
            referral_flag = prog_res.referral_flag

        new_prognosis = db_models.PrognosisResult(
            prediction_id=new_prediction.id,
            rest_weeks_min=rest_weeks_min,
            rest_weeks_max=rest_weeks_max,
            cast_type=cast_type,
            plaster_required=plaster_required,
            weight_bearing_status=weight_bearing_status,
            referral_flag=db_models.ReferralFlag(referral_flag),
            clinician_override=False
        )
        db.add(new_prognosis)
    except Exception as e:
        db.rollback()
        if saved_file_path.exists():
            os.remove(saved_file_path)
        if rel_heatmap_path:
            abs_heatmap = Path(rel_heatmap_path)
            if abs_heatmap.exists():
                os.remove(abs_heatmap)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AO guidelines prognosis calculation execution failed: {e}"
        )

    # Commit transactions
    db.commit()

    # Construct absolute heatmap URL for client dashboard
    base_url = str(request.base_url)
    heatmap_url = f"{base_url}{rel_heatmap_path}" if rel_heatmap_path else ""

    return {
        "scan_id": new_scan.id,
        "fracture_detected": inference_res.fracture_detected,
        "severity": inference_res.severity,
        "bone_affected": inference_res.bone_affected,
        "severity_confidence": inference_res.severity_confidence,
        "cast_type": cast_type,
        "rest_weeks_min": rest_weeks_min,
        "rest_weeks_max": rest_weeks_max,
        "plaster_required": plaster_required,
        "weight_bearing_status": weight_bearing_status,
        "referral_flag": referral_flag,
        "heatmap_url": heatmap_url,
        "original_file_path": rel_file_path
    }

@app.get("/scan/{scan_id}")
def get_scan(scan_id: UUID, request: Request, db: Session = Depends(get_db)):
    """Retrieves full details of a specific scan, prediction, and prognosis results."""
    scan = db.query(db_models.XrayScan).filter(db_models.XrayScan.id == scan_id).first()
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan record not found."
        )

    scan_data = {
        "scan_id": scan.id,
        "patient_id": scan.patient_id,
        "upload_timestamp": scan.upload_timestamp,
        "original_file_path": f"{request.base_url}{scan.original_file_path}",
        "bone_affected": scan.bone_affected,
        "image_quality_flag": scan.image_quality_flag,
        "dataset_source": scan.dataset_source.value,
        "prediction": None
    }

    if scan.prediction:
        pred = scan.prediction
        heatmap_url = f"{request.base_url}{pred.heatmap_path}" if pred.heatmap_path else ""
        
        scan_data["prediction"] = {
            "prediction_id": pred.id,
            "fracture_detected": pred.fracture_detected,
            "severity": pred.severity.value if pred.severity else "normal",
            "confidence_score": pred.confidence_score,
            "heatmap_url": heatmap_url,
            "model_version": pred.model_version,
            "prognosis": None
        }

        if pred.prognosis:
            prog = pred.prognosis
            scan_data["prediction"]["prognosis"] = {
                "prognosis_id": prog.id,
                "rest_weeks_min": prog.rest_weeks_min,
                "rest_weeks_max": prog.rest_weeks_max,
                "cast_type": prog.cast_type,
                "plaster_required": prog.plaster_required,
                "weight_bearing_status": prog.weight_bearing_status,
                "referral_flag": prog.referral_flag.value,
                "clinician_override": prog.clinician_override,
                "override_notes": prog.override_notes,
                "override_timestamp": prog.override_timestamp
            }

    return scan_data

@app.get("/patients/{patient_id}/history", response_model=schemas.PatientHistoryResponse)
def get_patient_history(patient_id: UUID, db: Session = Depends(get_db)):
    """Retrieves the patient history profile including all related X-ray scans, predictions, and prognoses."""
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Map model relationship directly using Pydantic serialization
    # Pydantic v2 maps the ORM fields using `from_attributes = True` configuration
    return {
        "patient_id": patient.id,
        "full_name": patient.full_name,
        "age": patient.age,
        "gender": patient.gender,
        "comorbidities": patient.comorbidities,
        "created_at": patient.created_at,
        "scans": patient.scans
    }

# ------------------------------------------------------------------------------
# Prognosis Override Endpoints
# ------------------------------------------------------------------------------

@app.patch("/prognosis/{prognosis_id}/override", response_model=schemas.PrognosisOverrideResponse)
def override_prognosis(
    prognosis_id: UUID, 
    override_in: schemas.PrognosisOverrideRequest, 
    db: Session = Depends(get_db)
):
    """
    Applies manual override to prognosis recommendations.
    Sets clinician_override to True, saves override notes containing the clinician 
    identifier, updates the timestamp, and returns the modified record.
    """
    prognosis = db.query(db_models.PrognosisResult).filter(
        db_models.PrognosisResult.id == prognosis_id
    ).first()

    if not prognosis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prognosis record with ID {prognosis_id} not found."
        )

    # Set override logs
    prognosis.clinician_override = True
    prognosis.override_notes = f"Override authorized by {override_in.clinician_override}. Notes: {override_in.override_notes}"
    prognosis.override_timestamp = datetime.utcnow()

    try:
        db.commit()
        db.refresh(prognosis)
        return prognosis
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error applying clinician override: {e}"
        )

@app.get("/prognosis/overrides", response_model=List[schemas.PrognosisOverrideResponse])
def get_all_overrides(db: Session = Depends(get_db)):
    """Retrieves all prognosis records containing clinician manual overrides."""
    overrides = db.query(db_models.PrognosisResult).filter(
        db_models.PrognosisResult.clinician_override == True
    ).all()
    return overrides

# ------------------------------------------------------------------------------
# Model Binding Setup (Ensuring pre-load reference is in state)
# ------------------------------------------------------------------------------
from inference import model as preloaded_model, device as preloaded_device

@app.on_event("startup")
def startup_bind_model():
    """Binds pre-loaded inference parameters to FastAPI app state."""
    app.state.model = preloaded_model
    app.state.device = preloaded_device
    print(f"[*] API Startup: Loaded multi-task FractureModel on {preloaded_device} successfully.")

if __name__ == "__main__":
    import uvicorn
    # Start main application on localhost:8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
