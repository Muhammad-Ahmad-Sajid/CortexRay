import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn.functional as F

# Ensure project root is in path for resolving src.* imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.model_training.model import FractureModel
from src.data_preparation.preprocess import get_inference_transform
from gradcam import generate_heatmap

from src.config import MODEL_CHECKPOINT_PATH

# Constants
CHECKPOINT_PATH = str(MODEL_CHECKPOINT_PATH)
SEVERITY_CLASSES = ['hairline', 'simple', 'displaced', 'comminuted']
BONE_CLASSES = ['distal_radius', 'clavicle', 'ankle', 'femur', 'humerus', 'metatarsal']

# ------------------------------------------------------------------------------
# Dataclass for structured inference results
# ------------------------------------------------------------------------------
@dataclass
class InferenceResult:
    fracture_detected: bool
    severity: str
    bone_affected: str
    severity_confidence: float
    bone_confidence: float
    heatmap_path: str

# ------------------------------------------------------------------------------
# Module-level Model Load (Loads once on import)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FractureModel(pretrained=False)

checkpoint_file = Path(CHECKPOINT_PATH)
if checkpoint_file.exists():
    print(f"[*] Inference Module: Loading pre-trained model from {checkpoint_file.absolute()}...")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Check if this is the old single-head checkpoint
        is_old_checkpoint = "backbone.fc.1.weight" in state_dict and "severity_head.0.weight" not in state_dict
        
        if is_old_checkpoint:
            print("[*] Inference Module: Detected old single-head model checkpoint. Adapting architecture...")
            model.severity_head = torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.Linear(2048, 4)
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                if k == "backbone.fc.1.weight":
                    new_state_dict["severity_head.1.weight"] = v
                elif k == "backbone.fc.1.bias":
                    new_state_dict["severity_head.1.bias"] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict, strict=False)
        print("[*] Inference Module: Model loaded successfully.")
    except Exception as e:
        print(f"[!] Inference Module Error loading state dict: {e}")
        print("[!] Running with randomly initialized weights.")
else:
    print(f"[!] Inference Module Warning: Checkpoint {checkpoint_file} not found.")
    print("[!] Running with randomly initialized weights for testing.")

model.to(device)
model.eval()

def run_inference(image_path: str) -> InferenceResult:
    """
    Performs end-to-end bone fracture classification and generates explainability overlays.
    
    Args:
        image_path (str): File path to the input grayscale X-ray image (PNG or JPG).
        
    Returns:
        InferenceResult: A structured dataclass containing detection outputs and heatmap paths.
    """
    # 1. Load and Preprocess Image
    # Reads the image from path and applies standard transforms (CLAHE, RGB, resize 224x224, normalization)
    # Import cv2 dynamically to keep core imports clean
    import cv2
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Contrast enhancement (matches preprocessing)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    transform = get_inference_transform()
    transformed = transform(image=img_resized)
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # 2. Run Forward Pass through multi-task model
    with torch.no_grad():
        severity_logits, bone_logits = model(input_tensor)
        
        # Apply softmax to obtain normalized probability distributions
        severity_probs = F.softmax(severity_logits, dim=1)
        bone_probs = F.softmax(bone_logits, dim=1)
        
        # Retrieve highest probability classes and confidences
        conf_sev, pred_sev = severity_probs.max(1)
        conf_bone, pred_bone = bone_probs.max(1)
        
        severity_confidence = round(float(conf_sev.item()), 2)
        bone_confidence = round(float(conf_bone.item()), 2)
        
        pred_sev_idx = int(pred_sev.item())
        pred_bone_idx = int(pred_bone.item())

    # 3. Determine if fracture is detected
    # If severity prediction confidence is low (< 0.30), we classify it as 'normal' (no fracture).
    # Since the 4-class classifier's random guess is 25%, a confidence >= 30% indicates a fracture prediction.
    if severity_confidence < 0.30:
        fracture_detected = False
        severity = "normal"
    else:
        fracture_detected = True
        severity = SEVERITY_CLASSES[pred_sev_idx]
        
    bone_affected = BONE_CLASSES[pred_bone_idx]

    # 4. Generate Grad-CAM Heatmap overlay
    try:
        heatmap_path = generate_heatmap(image_path, CHECKPOINT_PATH)
    except Exception as e:
        print(f"Warning: Grad-CAM heatmap generation failed during inference: {e}")
        heatmap_path = ""

    # 5. Return structured dataclass
    return InferenceResult(
        fracture_detected=fracture_detected,
        severity=severity,
        bone_affected=bone_affected,
        severity_confidence=severity_confidence,
        bone_confidence=bone_confidence,
        heatmap_path=heatmap_path
    )
