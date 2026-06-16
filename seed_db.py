import sys
from pathlib import Path

# Add the project root folder to the Python path to allow seamless imports of src.*
sys.path.append(str(Path(__file__).resolve().parent))

from src.database.connection import SessionLocal
from src.database.models import (
    Patient, 
    XrayScan, 
    FracturePrediction, 
    PrognosisResult,
    DatasetSource, 
    Severity, 
    ReferralFlag
)

def seed():
    # Instantiate database session
    db = SessionLocal()
    try:
        # Check if patient records exist to prevent duplicate seeding
        if db.query(Patient).count() > 0:
            print("Database already contains seeded patient records. Skipping seed process.")
            return

        print("Starting database seed process...")
        print("=" * 80)

        # ------------------------------------------------------------------------------
        # Patient 1: Teenager (14yo) with no comorbidities and a hairline fracture
        # ------------------------------------------------------------------------------
        p1 = Patient(
            full_name="Tommy Shelby",
            age=14,
            gender="Male",
            comorbidities=[]
        )
        db.add(p1)
        db.flush()  # Populate p1.id
        print(f"Inserted Patient: {p1.full_name} (Age: {p1.age})")

        s1 = XrayScan(
            patient_id=p1.id,
            original_file_path="uploads/tommy_shelby_xray.png",
            bone_affected="Radius",
            image_quality_flag="Good",
            dataset_source=DatasetSource.uploaded
        )
        db.add(s1)
        db.flush()
        print(f"  -> Inserted X-ray Scan for bone: {s1.bone_affected}")

        pred1 = FracturePrediction(
            scan_id=s1.id,
            fracture_detected=True,
            severity=Severity.hairline,
            confidence_score=0.88,
            heatmap_path="heatmaps/tommy_shelby_heatmap.png",
            model_version="v1.0.0"
        )
        db.add(pred1)
        db.flush()
        print(f"  -> Inserted Prediction: Fracture detected (Severity: {pred1.severity.value}, Confidence: {pred1.confidence_score})")

        prog1 = PrognosisResult(
            prediction_id=pred1.id,
            rest_weeks_min=3,
            rest_weeks_max=4,
            cast_type="Splint",
            plaster_required=False,
            weight_bearing_status="Full as tolerated",
            referral_flag=ReferralFlag.conservative
        )
        db.add(prog1)
        print(f"  -> Inserted Prognosis: Rest {prog1.rest_weeks_min}-{prog1.rest_weeks_max} weeks, Cast: {prog1.cast_type}")
        print("-" * 50)

        # ------------------------------------------------------------------------------
        # Patient 2: Young Adult (28yo) with no comorbidities and a simple fracture
        # ------------------------------------------------------------------------------
        p2 = Patient(
            full_name="Sarah Connor",
            age=28,
            gender="Female",
            comorbidities=[]
        )
        db.add(p2)
        db.flush()
        print(f"Inserted Patient: {p2.full_name} (Age: {p2.age})")

        s2 = XrayScan(
            patient_id=p2.id,
            original_file_path="mura_images/patient001/study1/scan.png",
            bone_affected="Humerus",
            image_quality_flag="Good",
            dataset_source=DatasetSource.MURA
        )
        db.add(s2)
        db.flush()
        print(f"  -> Inserted X-ray Scan for bone: {s2.bone_affected}")

        pred2 = FracturePrediction(
            scan_id=s2.id,
            fracture_detected=True,
            severity=Severity.simple,
            confidence_score=0.93,
            heatmap_path="heatmaps/sarah_connor_heatmap.png",
            model_version="v1.0.0"
        )
        db.add(pred2)
        db.flush()
        print(f"  -> Inserted Prediction: Fracture detected (Severity: {pred2.severity.value}, Confidence: {pred2.confidence_score})")

        prog2 = PrognosisResult(
            prediction_id=pred2.id,
            rest_weeks_min=6,
            rest_weeks_max=8,
            cast_type="Short Arm Plaster Cast",
            plaster_required=True,
            weight_bearing_status="Non-weight bearing",
            referral_flag=ReferralFlag.conservative
        )
        db.add(prog2)
        print(f"  -> Inserted Prognosis: Rest {prog2.rest_weeks_min}-{prog2.rest_weeks_max} weeks, Cast: {prog2.cast_type}")
        print("-" * 50)

        # ------------------------------------------------------------------------------
        # Patient 3: Middle-Aged (45yo) with Diabetes and a displaced femur fracture
        # ------------------------------------------------------------------------------
        p3 = Patient(
            full_name="Bruce Wayne",
            age=45,
            gender="Male",
            comorbidities=["Diabetes"]
        )
        db.add(p3)
        db.flush()
        print(f"Inserted Patient: {p3.full_name} (Age: {p3.age})")

        s3 = XrayScan(
            patient_id=p3.id,
            original_file_path="fracatlas_images/images/fractured/scan_002.png",
            bone_affected="Femur",
            image_quality_flag="Good",
            dataset_source=DatasetSource.FracAtlas
        )
        db.add(s3)
        db.flush()
        print(f"  -> Inserted X-ray Scan for bone: {s3.bone_affected}")

        pred3 = FracturePrediction(
            scan_id=s3.id,
            fracture_detected=True,
            severity=Severity.displaced,
            confidence_score=0.97,
            heatmap_path="heatmaps/bruce_wayne_heatmap.png",
            model_version="v1.0.0"
        )
        db.add(pred3)
        db.flush()
        print(f"  -> Inserted Prediction: Fracture detected (Severity: {pred3.severity.value}, Confidence: {pred3.confidence_score})")

        prog3 = PrognosisResult(
            prediction_id=pred3.id,
            rest_weeks_min=10,
            rest_weeks_max=12,
            cast_type="Long Leg Cast",
            plaster_required=True,
            weight_bearing_status="Non-weight bearing",
            referral_flag=ReferralFlag.surgical
        )
        db.add(prog3)
        print(f"  -> Inserted Prognosis: Rest {prog3.rest_weeks_min}-{prog3.rest_weeks_max} weeks, Cast: {prog3.cast_type}")
        print("-" * 50)

        # ------------------------------------------------------------------------------
        # Patient 4: Older Adult (62yo) with Osteoporosis and comminuted fracture
        # ------------------------------------------------------------------------------
        p4 = Patient(
            full_name="Ellen Ripley",
            age=62,
            gender="Female",
            comorbidities=["Osteoporosis"]
        )
        db.add(p4)
        db.flush()
        print(f"Inserted Patient: {p4.full_name} (Age: {p4.age})")

        s4 = XrayScan(
            patient_id=p4.id,
            original_file_path="uploads/ellen_ripley_xray.png",
            bone_affected="Tibia",
            image_quality_flag="Good",
            dataset_source=DatasetSource.uploaded
        )
        db.add(s4)
        db.flush()
        print(f"  -> Inserted X-ray Scan for bone: {s4.bone_affected}")

        pred4 = FracturePrediction(
            scan_id=s4.id,
            fracture_detected=True,
            severity=Severity.comminuted,
            confidence_score=0.99,
            heatmap_path="heatmaps/ellen_ripley_heatmap.png",
            model_version="v1.0.0"
        )
        db.add(pred4)
        db.flush()
        print(f"  -> Inserted Prediction: Fracture detected (Severity: {pred4.severity.value}, Confidence: {pred4.confidence_score})")

        prog4 = PrognosisResult(
            prediction_id=pred4.id,
            rest_weeks_min=12,
            rest_weeks_max=16,
            cast_type="Post-op Brace",
            plaster_required=False,
            weight_bearing_status="Non-weight bearing",
            referral_flag=ReferralFlag.surgical
        )
        db.add(prog4)
        print(f"  -> Inserted Prognosis: Rest {prog4.rest_weeks_min}-{prog4.rest_weeks_max} weeks, Cast: {prog4.cast_type}")
        print("-" * 50)

        # ------------------------------------------------------------------------------
        # Patient 5: Elderly (75yo) with both Diabetes and Osteoporosis
        # ------------------------------------------------------------------------------
        p5 = Patient(
            full_name="Arthur Dent",
            age=75,
            gender="Male",
            comorbidities=["Osteoporosis", "Diabetes"]
        )
        db.add(p5)
        db.flush()
        print(f"Inserted Patient: {p5.full_name} (Age: {p5.age})")

        s5 = XrayScan(
            patient_id=p5.id,
            original_file_path="uploads/arthur_dent_xray.png",
            bone_affected="Fibula",
            image_quality_flag="Good",
            dataset_source=DatasetSource.uploaded
        )
        db.add(s5)
        db.flush()
        print(f"  -> Inserted X-ray Scan for bone: {s5.bone_affected}")

        pred5 = FracturePrediction(
            scan_id=s5.id,
            fracture_detected=True,
            severity=Severity.simple,
            confidence_score=0.91,
            heatmap_path="heatmaps/arthur_dent_heatmap.png",
            model_version="v1.0.0"
        )
        db.add(pred5)
        db.flush()
        print(f"  -> Inserted Prediction: Fracture detected (Severity: {pred5.severity.value}, Confidence: {pred5.confidence_score})")

        prog5 = PrognosisResult(
            prediction_id=pred5.id,
            rest_weeks_min=8,
            rest_weeks_max=10,
            cast_type="Short Leg Cast",
            plaster_required=True,
            weight_bearing_status="Partial weight bearing",
            referral_flag=ReferralFlag.conservative
        )
        db.add(prog5)
        print(f"  -> Inserted Prognosis: Rest {prog5.rest_weeks_min}-{prog5.rest_weeks_max} weeks, Cast: {prog5.cast_type}")

        # Commit all transactions
        db.commit()
        print("=" * 80)
        print("Database seed completed successfully!")
        print("=" * 80)

    except Exception as e:
        db.rollback()
        print(f"ERROR: Database seeding encountered an exception: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
