[![CI Pipeline](https://github.com/Muhammad-Ahmad-Sajid/CortexRay/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Ahmad-Sajid/CortexRay/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# CortexRay
**AI-powered bone fracture detection and clinical prognosis system**

CortexRay is a comprehensive medical AI system designed to assist orthopedic specialists and radiologists in diagnosing bone fractures from X-ray imagery. Built around a custom two-stage trained ResNet-50 architecture, it not only detects fractures with high precision but also generates explainable AI heatmaps, formulates clinical prognoses, and compiles standardized PDF reports. This system provides an end-to-end clinical workflow bridging deep learning with practical, secure medical data management.

## 📸 Demo Screenshot
> 📸 Screenshot — add dashboard screenshot here

## 🌟 Key Features
- 🦴 **Fracture detection from X-ray images (98.15% accuracy):** Highly accurate multi-region bone fracture identification.
- 🧠 **ResNet-50 deep learning with two-stage training:** Robust pretraining on MURA followed by fine-tuning on FracAtlas.
- 🔥 **Grad-CAM heatmaps showing exact fracture location:** Visual, interpretable AI mapping highlighting clinical zones of interest.
- 📋 **Automated clinical PDF report generation:** Instant compilation of patient data, AI findings, and clinical recommendations.
- ⚕️ **Rule-based prognosis engine:** Formulates recommended rest duration, cast type, and specialist referral flags.
- 🔒 **JWT authentication with doctor and admin roles:** Secure, role-based access control for medical staff.
- 📊 **MLflow experiment tracking for all predictions:** Full observability over model confidence and inference timing.
- 🐳 **Fully Dockerized:** Runs effortlessly across environments with a single `docker-compose up` command.
- ✅ **CI/CD pipeline via GitHub Actions:** Automated testing and integration workflows.

## 🏗️ System Architecture

```text
X-ray Upload (Doctor)
        ↓
   FastAPI Backend (main.py)
        ↓
   ┌────────────────────────────┐
   │     Inference Pipeline     │
   │  preprocessing.py          │
   │  → ResNet-50 (stage2_best) │
   │  → Grad-CAM heatmap        │
   │  → Confidence thresholding │
   └────────────────────────────┘
        ↓
   ┌────────────────────────────┐
   │    Prognosis Engine        │
   │  prognosis_engine.py       │
   │  → AO Foundation guidelines│
   │  → Age + comorbidity mods  │
   └────────────────────────────┘
        ↓
   ┌────────────────────────────┐
   │    Data Layer              │
   │  PostgreSQL (pgAdmin 4)    │
   │  MLflow tracking           │
   │  PDF report generation     │
   └────────────────────────────┘
        ↓
   Clinical Dashboard (Doctor)
```

## 💻 Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **ML Framework** | PyTorch 2.x | Model training and inference |
| **Model** | ResNet-50 | Pretrained CNN backbone |
| **Explainability** | Grad-CAM | Fracture localization heatmaps |
| **Backend** | FastAPI | REST API and file serving |
| **Authentication** | JWT (python-jose) | Doctor and admin roles |
| **Database** | PostgreSQL 16 | Patient and scan storage |
| **ORM** | SQLAlchemy | Database models and queries |
| **ML Tracking** | MLflow | Experiment and inference logging |
| **PDF Generation** | ReportLab / WeasyPrint | Clinical report generation |
| **Containerization**| Docker + Compose | Deployment and portability |
| **CI/CD** | GitHub Actions | Automated testing pipeline |
| **Training Data** | MURA + FracAtlas | 44,000+ X-ray images |

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Fracture Detection Accuracy | 98.15% |
| Body Region Accuracy | 97.16% |
| Fracture Detection F1 | 0.98 |
| Best Epoch | 8 |
| Training Time | ~8 hours |
| Pretraining Dataset | MURA (40,005 images) |
| Fine-tuning Dataset | FracAtlas + Archive (combined) |
| Model Checkpoint Size | 270.4 MB |

## 📁 Project Structure

```text
CortexRay/
├── .env                        # Environment variables configuration
├── .gitignore                  # Git ignore rules
├── docker-compose.yml          # Docker services configuration
├── Dockerfile                  # API and inference container configuration
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── auth.py                     # JWT authentication logic
├── schemas.py                  # Pydantic validation models
├── inference.py                # ML model inference wrapper
├── prognosis_engine.py         # Clinical prognosis rules engine
├── report_generator.py         # PDF clinical report builder
├── src/
│   ├── database/
│   │   ├── connection.py       # SQLAlchemy engine and session
│   │   └── models.py           # Database ORM models
│   └── model_training/
│       ├── model.py            # ResNet-50 dual-head architecture
│       ├── prepare_mura.py     # Stage 1 data preparation
│       ├── train_stage1.py     # Stage 1 MURA pretraining
│       ├── prepare_stage2_data.py # Stage 2 FracAtlas data preparation
│       └── train_stage2.py     # Stage 2 fine-tuning
├── templates/
│   ├── index.html              # Main dashboard frontend
│   ├── history.html            # Patient history frontend
│   ├── style.css               # Frontend styling
│   └── report_template.html    # HTML template for PDF reports
├── checkpoints/
│   └── stage2_best.pth         # Trained model weights
├── uploads/                    # Directory for uploaded X-rays
├── heatmaps/                   # Directory for generated Grad-CAM images
└── reports/                    # Directory for generated PDF reports
```

## 🚀 Quick Start

### Prerequisites:
- Python 3.10+
- Docker and Docker Compose
- PostgreSQL 16 (via pgAdmin 4)
- CUDA-compatible GPU (recommended)

### Option A — Docker (recommended):
```bash
# Clone the repository
git clone https://github.com/Muhammad-Ahmad-Sajid/CortexRay.git
cd CortexRay

# Start the application using Docker Compose
docker-compose up --build -d
```

### Option B — Local development:
1. Clone the repo: `git clone https://github.com/Muhammad-Ahmad-Sajid/CortexRay.git`
2. Create virtual environment: `python -m venv myenv` and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Set up `.env` file (see template below)
5. Set up `fracture_db` in pgAdmin 4 (or local PostgreSQL) and run migrations/schemas
6. Start the FastAPI server: `uvicorn main:app --reload --port 8000`

## ⚙️ Environment Variables

Create a `.env` file in the root directory using the following template:

```env
# Database Connection string (Update with your credentials)
DATABASE_URL=postgresql://postgres:password@localhost:5432/fracture_db

# Security & Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# File Storage Directories
UPLOAD_FOLDER=uploads
HEATMAP_OUTPUT_FOLDER=heatmaps

# Model Configuration
MODEL_CHECKPOINT_PATH=checkpoints/stage2_best.pth

# Database Password (For docker-compose)
POSTGRES_PASSWORD=your-postgres-password
```

## 🗄️ Database Setup

1. Open pgAdmin 4 (or your preferred PostgreSQL client).
2. Connect to your local server and create a new database named `fracture_db`.
3. Open the Query Tool for `fracture_db` and run the standard SQLAlchemy schema creation logic (automatically handled by `main.py` if tables do not exist).
4. Run the database seeding script to populate test data and admin credentials:
   ```bash
   python seed_db.py
   ```

## 🏋️ Training the Model

CortexRay utilizes a sophisticated two-stage training methodology. 

### Stage 1 — Pretrain on MURA:
1. Download the MURA dataset via Redivis or Stanford ML group.
2. Prepare the dataset:
   ```bash
   python src/model_training/prepare_mura.py
   ```
3. Run Stage 1 training:
   ```bash
   python src/model_training/train_stage1.py
   ```
   *Expected output: Validation accuracy ~83%*

### Stage 2 — Fine-tune on FracAtlas:
1. Download the FracAtlas dataset from Kaggle.
2. Prepare and combine the dataset:
   ```bash
   python src/model_training/prepare_stage2_data.py
   ```
3. Run Stage 2 fine-tuning:
   ```bash
   python src/model_training/train_stage2.py
   ```
   *Expected output: Fracture accuracy ~98%, Body region accuracy ~97%*

## 🔌 API Reference

Full interactive API documentation is available at `http://localhost:8000/docs` once the server is running.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | Admin | Register new user |
| POST | `/auth/login` | Public | Login and get JWT token |
| GET | `/auth/me` | Doctor | Get current user info |
| POST | `/patients/` | Doctor | Register new patient |
| POST | `/scan/upload` | Doctor | Upload X-ray and run analysis |
| GET | `/scan/{id}` | Doctor | Get scan results |
| GET | `/scan/{id}/report` | Doctor | Download PDF report |
| GET | `/patients/{id}/history` | Doctor | Get patient scan history |
| PATCH | `/prognosis/{id}/override` | Doctor | Submit clinician correction |
| GET | `/prognosis/overrides` | Admin | View all clinician overrides |
| GET | `/health` | Public | System health check |

## ⚖️ Confidence Thresholding

The system strictly categorizes predictions to ensure clinical safety and reliability:

| Flag | Threshold | Meaning | Action |
|------|-----------|---------|--------|
| **clear** | ≥ 80% | High confidence prediction | Full prognosis + PDF report generated |
| **low_confidence** | 60–79% | Uncertain prediction | Result shown, repeat scan recommended |
| **inconclusive** | < 60% | Too uncertain | No prognosis, manual radiologist review required |

## ⚠️ Clinical Disclaimer

> ⚠️ CortexRay is an AI-assisted clinical decision support tool intended for use by qualified medical professionals only. It does not replace radiologist diagnosis or clinical judgment. All AI predictions must be reviewed and confirmed by a licensed clinician before any treatment decision is made. The authors accept no liability for clinical decisions made based on this system's output.

## 📚 Dataset Citations

- **MURA:** Rajpurkar, P., Irvin, J., Bagul, A., Ding, D., Duan, T., Mehta, H., ... & Ng, A. Y. (2017). MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs. [arXiv:1712.06957](https://arxiv.org/abs/1712.06957)
- **FracAtlas:** Abedalla, A., Abdullah, M., Al-Ayyoub, M. et al. The FracAtlas dataset for fracture detection in musculoskeletal radiographs. Sci Data 10, 368 (2023). [https://doi.org/10.1038/s41597-023-02432-4](https://doi.org/10.1038/s41597-023-02432-4)

## 📄 License

This project is licensed under the **MIT License**.

## 👤 Author

**Muhammad Ahmad Sajid**  
BSAI Student — Ghulam Ishaq Khan Institute (GIKI)  
GitHub: [Muhammad-Ahmad-Sajid](https://github.com/Muhammad-Ahmad-Sajid)  
Project: [CortexRay](https://github.com/Muhammad-Ahmad-Sajid/CortexRay)
