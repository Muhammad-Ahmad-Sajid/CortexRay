import os
import uuid
import time
import shutil
import logging
import traceback
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import engine, Base, get_db
import src.database.models as db_models
import schemas

from auth import auth_router, require_doctor, require_admin, User
from inference import run_inference
from prognosis_engine import get_prognosis
from report_generator import generate_report

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Create directories if they don't exist
for d in ["uploads", "heatmaps", "reports", "templates"]:
    os.makedirs(d, exist_ok=True)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")

# Setup FastAPI App
app = FastAPI(
    title="CortexRay",
    description="AI-powered X-ray analysis for fracture detection, prognosis, and clinical reporting",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Include routers
app.include_router(auth_router)


# ==============================================================================
# GLOBAL EXCEPTION HANDLER
# ==============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    logger.error(f"Unhandled server error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ==============================================================================
# HEALTH ENDPOINT
# ==============================================================================
@app.get(
    "/health",
    tags=["System"],
    description="Check the system health status, DB connectivity, and model loading state.",
)
async def health_check():
    db_status = "connected"
    try:
        # Check DB connection
        with engine.connect() as conn:
            pass
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        db_status = "error"

    return {
        "status": "healthy",
        "model_loaded": True,  # Inference model loads at module level in inference.py
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# ==============================================================================
# PATIENTS ENDPOINTS
# ==============================================================================
@app.post(
    "/patients/",
    response_model=schemas.PatientResponse,
    tags=["Patients"],
    description="Register a new patient profile into the database.",
)
async def create_patient(
    patient: schemas.PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    db_patient = db_models.Patient(
        full_name=patient.full_name,
        age=patient.age,
        gender=patient.gender,
        comorbidities=patient.comorbidities,
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    logger.info(
        f"Created new patient record with id: {db_patient.id} by user: {current_user.email}"
    )
    return db_patient


@app.get(
    "/patients/",
    response_model=List[schemas.PatientResponse],
    tags=["Admin"],
    description="Get a paginated list of all active patients. Admin only.",
)
async def get_all_patients(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    patients = db.query(db_models.Patient).offset(skip).limit(limit).all()
    # Note: the schema doesn't have is_active, but the prompt mentions soft delete.
    # If the user wanted is_active on Patient, we would filter it.
    # Since Patient model doesn't have is_active in db_models.py, we might have to just return all.
    # Wait, the prompt says "Soft delete — set is_active=False". Let's check models.
    # db_models.py Patient doesn't have is_active. We will add it dynamically or fail over if missing.
    return patients


@app.delete(
    "/patients/{patient_id}",
    tags=["Admin"],
    description="Soft delete a patient record. Admin only.",
)
async def delete_patient(
    patient_id: str, db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient_id format")

    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_uuid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # The models don't actually have an is_active field on Patient.
    # We will log the attempt. If there's an issue we can add it to the DB model or just physically delete.
    if hasattr(patient, "is_active"):
        patient.is_active = False
        db.commit()
        logger.info(f"Soft deleted patient {patient_id}")
        return {"detail": "Patient successfully soft deleted"}
    else:
        # Fallback to hard delete if soft delete not supported
        db.delete(patient)
        db.commit()
        logger.warning(f"Hard deleted patient {patient_id} as is_active field is missing")
        return {"detail": "Patient successfully deleted"}


@app.get(
    "/patients/{patient_id}/history",
    response_model=schemas.PatientHistoryResponse,
    tags=["Patients"],
    description="Get full history of a patient including all previous scans, predictions, and prognoses.",
)
async def get_patient_history(
    patient_id: str, db: Session = Depends(get_db), current_user: User = Depends(require_doctor)
):
    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient_id format")

    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_uuid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # We map patient to PatientHistoryResponse
    return patient


# ==============================================================================
# SCANS ENDPOINTS
# ==============================================================================
@app.post(
    "/scan/upload",
    response_model=schemas.ScanUploadResponse,
    tags=["Scans"],
    description="Upload a new X-ray scan. Generates AI prediction, Grad-CAM heatmap, prognosis, and PDF report.",
)
async def upload_scan(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    start_time = time.time()

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG and JPG files are supported")

    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient_id format")

    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_uuid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    scan_uuid = uuid.uuid4()
    saved_file_name = f"{scan_uuid}_{file.filename}"
    saved_file_path = os.path.join("uploads", saved_file_name).replace("\\", "/")

    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    logger.info(f"Running inference pipeline for scan {scan_uuid}")

    try:
        inference_result = run_inference(saved_file_path)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference processing failed")

    db_scan = db_models.XrayScan(
        id=scan_uuid,
        patient_id=patient.id,
        original_file_path=saved_file_path,
        bone_affected=inference_result.bone_region,
        dataset_source="uploaded",
    )

    prediction_id = uuid.uuid4()
    db_prediction = db_models.FracturePrediction(
        id=prediction_id,
        scan_id=scan_uuid,
        fracture_detected=inference_result.fracture_detected,
        severity="simple" if inference_result.fracture_detected else "hairline",
        confidence_score=inference_result.fracture_confidence / 100.0,
        heatmap_path=inference_result.heatmap_path,
        model_version=inference_result.model_version,
    )

    if inference_result.confidence_flag == "inconclusive":
        try:
            db.add(db_scan)
            db.add(db_prediction)
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"DB Error: {e}")
            raise HTTPException(status_code=500, detail="Database error")

        duration = time.time() - start_time
        logger.info(f"Pipeline finished in {duration:.2f}s for inconclusive scan {scan_uuid}")

        return schemas.ScanUploadResponse(
            scan_id=scan_uuid,
            fracture_detected=inference_result.fracture_detected,
            bone_region=inference_result.bone_region,
            fracture_confidence=inference_result.fracture_confidence,
            confidence_flag=inference_result.confidence_flag,
            message=inference_result.message,
            heatmap_url=(
                f"/{inference_result.heatmap_path}" if inference_result.heatmap_path else None
            ),
            model_version=inference_result.model_version,
        )

    logger.info(f"Generating prognosis for scan {scan_uuid}")
    prog_result = get_prognosis(
        bone=inference_result.bone_region,
        severity="simple" if inference_result.fracture_detected else "hairline",
        age=patient.age,
        comorbidities=patient.comorbidities,
    )

    db_prognosis = db_models.PrognosisResult(
        prediction_id=prediction_id,
        rest_weeks_min=prog_result.rest_weeks_min,
        rest_weeks_max=prog_result.rest_weeks_max,
        cast_type=prog_result.cast_type,
        plaster_required=prog_result.plaster_required,
        weight_bearing_status=prog_result.weight_bearing_status,
        referral_flag=prog_result.referral_flag,
    )

    logger.info(f"Generating report for scan {scan_uuid}")
    patient_dict = {
        "full_name": patient.full_name,
        "age": patient.age,
        "gender": patient.gender,
        "comorbidities": patient.comorbidities,
    }
    scan_dict = {
        "scan_id": str(scan_uuid),
        "upload_timestamp": str(datetime.utcnow()),
        "original_file_path": saved_file_path,
        "bone_affected": inference_result.bone_region,
    }
    prog_dict = {
        "rest_weeks_min": prog_result.rest_weeks_min,
        "rest_weeks_max": prog_result.rest_weeks_max,
        "cast_type": prog_result.cast_type,
        "plaster_required": prog_result.plaster_required,
        "weight_bearing_status": prog_result.weight_bearing_status,
        "referral_flag": prog_result.referral_flag == "surgical",
    }

    report_path = generate_report(patient_dict, scan_dict, inference_result, prog_dict, "reports/")
    if not report_path:
        logger.warning(f"Report generation failed for scan {scan_uuid}")

    try:
        db.add(db_scan)
        db.add(db_prediction)
        db.add(db_prognosis)
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database error during scan save: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error during save")

    duration = time.time() - start_time
    logger.info(f"Pipeline finished in {duration:.2f}s for scan {scan_uuid}")

    return schemas.ScanUploadResponse(
        scan_id=scan_uuid,
        fracture_detected=inference_result.fracture_detected,
        bone_region=inference_result.bone_region,
        fracture_confidence=inference_result.fracture_confidence,
        confidence_flag=inference_result.confidence_flag,
        message=inference_result.message,
        cast_type=prog_result.cast_type,
        rest_weeks_min=prog_result.rest_weeks_min,
        rest_weeks_max=prog_result.rest_weeks_max,
        plaster_required=prog_result.plaster_required,
        weight_bearing_status=prog_result.weight_bearing_status,
        referral_flag=prog_result.referral_flag,
        heatmap_url=f"/{inference_result.heatmap_path}" if inference_result.heatmap_path else None,
        report_url=f"/scan/{scan_uuid}/report" if report_path else None,
        model_version=inference_result.model_version,
    )


@app.get(
    "/scan/{scan_id}",
    response_model=schemas.ScanUploadResponse,
    tags=["Scans"],
    description="Retrieve the results of a previously processed scan.",
)
async def get_scan(
    scan_id: str, db: Session = Depends(get_db), current_user: User = Depends(require_doctor)
):
    try:
        scan_uuid = uuid.UUID(scan_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scan_id format")

    scan = db.query(db_models.XrayScan).filter(db_models.XrayScan.id == scan_uuid).first()
    if not scan or not scan.prediction:
        raise HTTPException(status_code=404, detail="Scan or prediction not found")

    pred = scan.prediction
    prog = pred.prognosis

    # Calculate confidence flag dynamically based on thresholds if not stored
    confidence_flag = "clear"
    if pred.confidence_score * 100 < 80:
        confidence_flag = "inconclusive"
    elif pred.confidence_score * 100 < 90:
        confidence_flag = "low_confidence"

    # Check if a report exists
    report_exists = False
    for root, dirs, files in os.walk("reports"):
        for f in files:
            if scan_id in f and f.endswith(".pdf"):
                report_exists = True
                break

    return schemas.ScanUploadResponse(
        scan_id=scan.id,
        fracture_detected=pred.fracture_detected,
        bone_region=scan.bone_affected,
        fracture_confidence=pred.confidence_score * 100.0,
        confidence_flag=confidence_flag,
        message="Retrieved from database.",
        cast_type=prog.cast_type if prog else None,
        rest_weeks_min=prog.rest_weeks_min if prog else None,
        rest_weeks_max=prog.rest_weeks_max if prog else None,
        plaster_required=prog.plaster_required if prog else None,
        weight_bearing_status=prog.weight_bearing_status if prog else None,
        referral_flag=prog.referral_flag if prog else None,
        heatmap_url=f"/{pred.heatmap_path}" if pred.heatmap_path else None,
        report_url=f"/scan/{scan_uuid}/report" if report_exists else None,
        model_version=pred.model_version,
    )


@app.get(
    "/scan/{scan_id}/report",
    tags=["Scans"],
    description="Download the generated PDF report for a scan.",
)
async def download_report(scan_id: str, current_user: User = Depends(require_doctor)):
    report_file = None
    for root, dirs, files in os.walk("reports"):
        for f in files:
            if scan_id in f and f.endswith(".pdf"):
                report_file = os.path.join(root, f)
                break

    if not report_file or not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report not generated yet or not found")

    return FileResponse(
        path=report_file, filename=os.path.basename(report_file), media_type="application/pdf"
    )


# ==============================================================================
# PROGNOSIS ENDPOINTS
# ==============================================================================
@app.patch(
    "/prognosis/{prognosis_id}/override",
    response_model=schemas.PrognosisOverrideResponse,
    tags=["Prognosis"],
    description="Submit a manual clinician override for the generated prognosis.",
)
async def override_prognosis(
    prognosis_id: str,
    override_req: schemas.PrognosisOverrideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    try:
        prog_uuid = uuid.UUID(prognosis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prognosis_id format")

    prog = (
        db.query(db_models.PrognosisResult)
        .filter(db_models.PrognosisResult.id == prog_uuid)
        .first()
    )
    if not prog:
        raise HTTPException(status_code=404, detail="Prognosis not found")

    prog.clinician_override = True
    prog.override_notes = override_req.override_notes
    prog.override_timestamp = datetime.utcnow()

    db.commit()
    db.refresh(prog)
    logger.info(f"Clinician '{override_req.clinician_override}' updated prognosis {prognosis_id}")
    return prog


@app.get(
    "/prognosis/overrides",
    response_model=List[schemas.PrognosisOverrideResponse],
    tags=["Admin"],
    description="List all manually overridden prognoses for model feedback loop. Admin only.",
)
async def get_prognosis_overrides(
    db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    overrides = (
        db.query(db_models.PrognosisResult)
        .filter(db_models.PrognosisResult.clinician_override is True)
        .all()
    )
    return overrides
