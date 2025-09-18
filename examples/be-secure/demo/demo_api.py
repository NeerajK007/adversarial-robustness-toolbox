import os
import io
import json
import base64
import logging
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
from demo_attack import generate_adversarial  # your attack module

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Model registry for demo types
# -------------------------------
MODEL_REGISTRY = {
    "Tile-defect-misclassification": {
        "model_fn": models.mobilenet_v2,
        "num_classes": 2,
        "weights_path": "weights/mobilenetv2_mvtec.pth"
    },
    "Indian-trafic-signal-misclassification": {
        "model_fn": models.mobilenet_v2,
        "num_classes": 4,
        "weights_path": "weights/20250918_091130_mobilenetv2_traffic_signs.pth"
    }
    # future demos can be added here
}

# -------------------------------
# Load model helper
# -------------------------------
def load_demo_model(demo_type: str, device, variant="normal"):
    logging.info(f"demo_type: {demo_type}")
    if demo_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown demo_type: {demo_type}")
    
    model_registry = MODEL_REGISTRY[demo_type]
     # default path
    weights_path = model_registry["weights_path"]
    # switch only for adv-trained variant
    if variant == "adv_trained" and demo_type == "Indian-trafic-signal-misclassification":
        weights_path = "weights/20250911_121629_mobilenetv2_traffic_signs_AdvTrained.pth"

    
    # Load base model
    model = model_registry["model_fn"](weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    
    # Adjust classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, model_registry["num_classes"])
    
    # Load finetuned weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    logging.info(f"Loaded {demo_type} model with weights: {weights_path}")
    return model

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Adversarial Demo API")

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://art.o31e.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    path = os.path.join(os.path.dirname(__file__), "demo_ui.html")
    return FileResponse(path)

# -------------------------------
# Predict endpoint
# -------------------------------
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    demo_type: str = Form(...),
    variant: str = Form("normal")
    ):
    try:
        logging.info(f"/predict calling...")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_demo_model(demo_type, device, variant=variant)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        
        # Map label for different demos
        if demo_type == "Tile-defect-misclassification":
            label = "good" if pred.item() == 0 else "defective"
        elif demo_type == "Indian-trafic-signal-misclassification":
            #class_labels = ["CROSS_ROAD", "FALLING_ROCKS", "NO_ENTRY", "PEDESTRIAN_CROSSING", "SCHOOL_AHEAD", "SPEED_LIMIT_70", "SPEED_LIMIT_80", "STOP"]
            class_labels = ["SCHOOL_AHEAD", "SPEED_LIMIT_70", "SPEED_LIMIT_80", "STOP"]
            logging.info(f"pred: {pred.item()}, label: {class_labels[pred.item()]}")
            label = class_labels[pred.item()]
        else:
            label = str(pred.item())
        
        return JSONResponse(content={"label": label, "confidence": float(confidence.item())})
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------
# Adversarial attack endpoint
# -------------------------------
@app.post("/attack")
async def generate_adv_image(
    file: UploadFile = File(...),
    demo_type: str = Form(...),
    attack: str = Form("fgsm"),
    eps: Optional[float] = Form(None),
    eps_step: Optional[float] = Form(None),
    max_iter: Optional[int] = Form(None),
    targeted: Optional[str] = Form(None),
    num_random_init: Optional[int] = Form(None),
    confidence: Optional[float] = Form(None),
    learning_rate: Optional[float] = Form(None),
    binary_search_steps: Optional[int] = Form(None),
    initial_const: Optional[float] = Form(None),
):
    start_ts = datetime.now().isoformat()
    logging.info(f"[ATTACK_API] Start: {start_ts} attack={attack}, demo_type={demo_type}")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_demo_model(demo_type, device)
        
        # Collect attack parameters dynamically
        attack_params = {}
        if eps is not None: attack_params["eps"] = float(eps)
        if eps_step is not None: attack_params["eps_step"] = float(eps_step)
        if max_iter is not None: attack_params["max_iter"] = int(max_iter)
        if targeted is not None: attack_params["targeted"] = targeted
        if num_random_init is not None: attack_params["num_random_init"] = int(num_random_init)
        if confidence is not None: attack_params["confidence"] = float(confidence)
        if learning_rate is not None: attack_params["learning_rate"] = float(learning_rate)
        if binary_search_steps is not None: attack_params["binary_search_steps"] = int(binary_search_steps)
        if initial_const is not None: attack_params["initial_const"] = float(initial_const)
        
        # Generate adversarial image
        result = generate_adversarial(
            image, model, device, attack_name=attack, attack_params=attack_params
        )
        adv_image = result["adversarial_image"]

        # Encode to base64
        buf = io.BytesIO()
        adv_image.save(buf, format="PNG")
        adv_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        end_ts = datetime.now().isoformat()
        logging.info(f"[ATTACK_API] End: {end_ts} (Duration: {datetime.fromisoformat(end_ts) - datetime.fromisoformat(start_ts)})")
        
        return JSONResponse(content={
            "attack": attack,
            "adv_image_base64": adv_base64
        })
    
    except Exception as e:
        logging.error(f"Adversarial image generation error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------
# Serve demo UI
# -------------------------------
@app.get("/demo_ui.html")
def serve_ui():
    path = os.path.join(os.path.dirname(__file__), "demo_ui.html")
    return FileResponse(path)
