from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from datetime import datetime
from typing import List, Optional

# ------------------------------------------------------------------------------
# Patient Schemas
# ------------------------------------------------------------------------------
class PatientCreate(BaseModel):
    """Schema for registering a new patient."""
    full_name: str = Field(..., description="Full legal name of the patient", min_length=2, max_length=255)
    age: int = Field(..., description="Age in years")
    gender: str = Field(..., description="Gender of the patient (e.g. Male, Female, Other)", min_length=2, max_length=50)
    comorbidities: List[str] = Field(default_factory=list, description="Array of health comorbidities")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Pydantic V2 validator checking that age is between 1 and 120."""
        if not (1 <= v <= 120):
            raise ValueError("Age must be between 1 and 120")
        return v

class PatientResponse(BaseModel):
    """Schema returning registered patient details."""
    id: UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

# ------------------------------------------------------------------------------
# Scan Upload & Inference Schemas
# ------------------------------------------------------------------------------
class ScanUploadResponse(BaseModel):
    """Schema returning the direct diagnostic outputs of an X-ray upload."""
    scan_id: UUID
    fracture_detected: bool
    severity: str
    bone_affected: str
    severity_confidence: float
    cast_type: str
    rest_weeks_min: int
    rest_weeks_max: int
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    heatmap_url: str
    original_file_path: str

    @field_validator('severity_confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Pydantic V2 validator checking that model confidence is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

# ------------------------------------------------------------------------------
# Prognosis Override Schemas
# ------------------------------------------------------------------------------
class PrognosisOverrideRequest(BaseModel):
    """Schema for clinician manual overrides of recovery recommendations."""
    clinician_override: str = Field(..., description="Name or identifier of the clinician authorizing the override", min_length=2)
    override_notes: str = Field(..., description="Clinical justification notes for the override", min_length=5)

class PrognosisOverrideResponse(BaseModel):
    """Schema returning the updated prognosis details after override."""
    id: UUID
    prediction_id: UUID
    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    clinician_override: bool
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]

    class Config:
        from_attributes = True

# ------------------------------------------------------------------------------
# Patient History & Diagnostics Detail Schemas
# ------------------------------------------------------------------------------
class PrognosisDetail(BaseModel):
    id: UUID
    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    clinician_override: bool
    override_notes: Optional[str] = None
    override_timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True

class PredictionDetail(BaseModel):
    id: UUID
    fracture_detected: bool
    severity: Optional[str] = None
    confidence_score: float
    heatmap_path: Optional[str] = None
    model_version: str
    prognosis: Optional[PrognosisDetail] = None

    class Config:
        from_attributes = True

class ScanDetail(BaseModel):
    id: UUID
    upload_timestamp: datetime
    original_file_path: str
    bone_affected: str
    image_quality_flag: str
    dataset_source: str
    prediction: Optional[PredictionDetail] = None

    class Config:
        from_attributes = True

class PatientHistoryResponse(BaseModel):
    """Schema returning patient details combined with complete diagnostic scan history."""
    patient_id: UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime
    scans: List[ScanDetail]

    class Config:
        from_attributes = True
