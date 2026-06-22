# Bone Fracture Detection & Prognosis System

[![CI Pipeline](https://github.com/Muhammad-Ahmad-Sajid/CortexRay/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Ahmad-Sajid/CortexRay/actions/workflows/ci.yml)
An AI-powered clinical decision support system that detects bone fractures from X-ray scans, classifies their severity level, identifies the affected bone, and calculates patient-specific recovery guidelines based on the AO Foundation orthopedic healing principles. The system includes a dual-head ResNet-50 deep learning model, a robust rules-based prognosis engine, a PostgreSQL database for persistent logging, and a clinical dashboard displaying comparative explainability heatmaps (Grad-CAM) alongside a complete, searchable history timeline with clinician override support.

---

## Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch, Torchvision | Dual-head ResNet-50 backbone for multi-task vision classification |
| **Backend API** | FastAPI, Uvicorn | High-performance, asynchronous REST API with validation |
| **Database** | PostgreSQL, SQLAlchemy | Relational storage for patients, scan metadata, and prognoses |
| **Frontend** | HTML5, CSS3, Vanilla JS | Clean clinical dashboard layout with responsive design |
| **Explainability** | Grad-CAM | Saliency maps overlaying deep neural network attention areas |
| **Datasets** | MURA, FracAtlas | Standard datasets used for pretraining and fine-tuning |

---

## System Architecture

Below is the high-level workflow mapping out the diagnostics and clinical override pipelines:

```text
                  +----------------------------------------------+
                  |              Web Dashboard UI                |
                  +-------+------------------------------^-------+
                          | (1) Upload X-Ray             | (6) Show Results
                          |                              |     & Heatmap
                          v                              |
            +-------------+-------------+                |
            |     FastAPI Server        +----------------+
            +-------------+-------------+
                          |
                          | (2) Grayscale / CLAHE / Resize
                          v
            +-------------+-------------+
            |  Image Preprocessing (CV2) |
            +-------------+-------------+
                          |
                          | (3) Normalization & Tensor
                          v
            +-------------+-------------+
            |  ResNet-50 Classifier (PyTorch) +--------+
            +-------------+-------------+              | (4) Predicts:
                          |                            |     - Bone Affected
                          |                            |     - Severity Level
                          v                            v
            +-------------+----------------------------+--+
            |          Prognosis Engine (AO Rules)         |
            |   (Modifiers: Age, Osteoporosis, Diabetes)   |
            +-------------+-------------------------------+
                          |
                          | (5) Save Diagnostics Logs & Results
                          v
            +-------------+-------------+
            |     PostgreSQL Database    |
            +---------------------------+
```

---

## Datasets & Model Training

The model is trained in a **two-stage training process**:

1. **Stage 1 (Pretraining)**: Backbone training on the **MURA** (Musculoskeletal Radiographs) dataset for binary normal/abnormal classification.
2. **Stage 2 (Fine-Tuning)**: Multi-task training on **FracAtlas** for simultaneous 4-class severity prediction (hairline, simple, displaced, comminuted) and 6-class bone type classification (distal_radius, clavicle, ankle, femur, humerus, metatarsal).

### Download Instructions

#### 1. MURA Dataset
- **Description**: Musculoskeletal Radiographs containing ~40,000 images from Stanford AIMI.
- **Download**: Access the dataset via the Stanford AIMI group website or fetch it using the Redivis API:
  ```bash
  pip install redivis
  redivis dataset stanford_aimi.mura:v1_1 download ./data/mura
  ```
- **Preprocessing**: Run `python prepare_mura.py` to organize images and create `mura_metadata.csv`.

#### 2. FracAtlas Dataset
- **Description**: 4,083 orthopedic X-ray scans with bone fracture annotations.
- **Download**: Download the dataset from Kaggle [FracAtlas Dataset](https://www.kaggle.com/datasets/vuppalaadithyasri/fracatlas).
- **Preprocessing**: Run `python prepare_fracatlas.py` to compile bounding box details and generate `fracatlas_labels.csv`.

### Training Command Sequences
- **Stage 1 (MURA Backbone Pretraining)**:
  ```bash
  python train_stage1.py
  ```
- **Stage 2 (FracAtlas Multi-Task Fine-Tuning)**:
  ```bash
  python train_stage2.py
  ```

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/bone-fracture-prognosis.git
cd bone-fracture-prognosis
```

### 2. Configure Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows (Powershell)
.\venv\Scripts\Activate.ps1
# On macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Initialize PostgreSQL Database
1. Open pgAdmin 4 or connect to your local PostgreSQL instance via terminal.
2. Execute the database initialization command:
   ```sql
   CREATE DATABASE fracture_db;
   ```
3. Run the schema creation script [database.sql](file:///d:/X-ray%20ML%20Model/database.sql) to set up all tables (`patients`, `xray_scans`, `fracture_predictions`, `prognosis_results`).

### 4. Setup Environment Variables
1. Copy the `.env.template` file to `.env`:
   ```bash
   cp .env.template .env
   ```
2. Update the credentials in `.env` to match your local database settings:
   ```text
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/fracture_db
   ```

### 5. Seed Test Patients & Start server
1. Seed the database with 5 initial mock patient profiles:
   ```bash
   python seed_db.py
   ```
2. Start the FastAPI application:
   ```bash
   python main.py
   ```
3. Navigate to `http://localhost:8000/` to open the main registration and scan analysis dashboard, and `http://localhost:8000/history` to view patient diagnostic logs.

---

## API Demonstration & Examples

### 1. Register Patient (`POST /patients/`)

**Request**:
```bash
curl -X POST "http://localhost:8000/patients/" \
     -H "Content-Type: application/json" \
     -d '{
       "full_name": "Arthur Dent",
       "age": 42,
       "gender": "Male",
       "comorbidities": ["Osteoporosis"]
     }'
```

**Response**:
```json
{
  "id": "2ac23ed6-4fc9-439b-9221-9a48d6558c89",
  "full_name": "Arthur Dent",
  "age": 42,
  "gender": "Male",
  "comorbidities": [
    "Osteoporosis"
  ],
  "created_at": "2026-06-05T18:32:00"
}
```

### 2. Upload X-Ray Scan (`POST /scan/upload`)

**Request**:
```bash
curl -X POST "http://localhost:8000/scan/upload" \
     -F "patient_id=2ac23ed6-4fc9-439b-9221-9a48d6558c89" \
     -F "file=@/path/to/distal_radius_xray.png"
```

**Response**:
```json
{
  "scan_id": "8b51ea1d-2856-4c48-b4b3-d021625f18c6",
  "fracture_detected": true,
  "severity": "simple",
  "bone_affected": "distal_radius",
  "severity_confidence": 0.942,
  "cast_type": "Short Arm Fiberglass Cast",
  "rest_weeks_min": 7,
  "rest_weeks_max": 9,
  "plaster_required": true,
  "weight_bearing_status": "Upper extremity: Non-weight-bearing (No lifting/pushing)",
  "referral_flag": "conservative",
  "heatmap_url": "http://localhost:8000/heatmaps/heatmap_8b51ea1d.png"
}
```

---

## Model Performance Metrics

Below is a placeholder table representing model classification validation targets to compile after full stage 1 and stage 2 training runs:

| Output Head | Class Name | Precision | Recall | F1-Score | Accuracy (Overall) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Severity Classifier** | Hairline | *[0.85]* | *[0.81]* | *[0.83]* | *[88.2%]* |
| | Simple | *[0.89]* | *[0.92]* | *[0.90]* | |
| | Displaced | *[0.87]* | *[0.85]* | *[0.86]* | |
| | Comminuted | *[0.91]* | *[0.88]* | *[0.89]* | |
| **Bone Classifier** | Distal Radius | *[0.94]* | *[0.96]* | *[0.95]* | *[94.7%]* |
| | Clavicle | *[0.92]* | *[0.90]* | *[0.91]* | |
| | Ankle | *[0.95]* | *[0.93]* | *[0.94]* | |
| | Femur | *[0.97]* | *[0.96]* | *[0.96]* | |
| | Humerus | *[0.91]* | *[0.93]* | *[0.92]* | |
| | Metatarsal | *[0.93]* | *[0.90]* | *[0.91]* | |

---

## Clinical Disclaimer

> [!CAUTION]
> **Clinical Decision Support Tool Only**  
> This software is research-grade and intended solely to provide diagnostic and prognosis recommendations as decision-support assistance for qualified medical clinicians. It is NOT FDA-approved, CE-certified, or cleared for autonomous medical diagnoses. All diagnostic classifications, explainability heatmaps, and recovery guidelines must be reviewed, verified, and signed off by a qualified radiologist or orthopedic surgeon.
