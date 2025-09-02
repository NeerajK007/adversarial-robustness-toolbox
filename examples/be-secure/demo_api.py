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
from fastapi.responses import JSONResponse
from demo_attack import generate_adversarial  # your attack module
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional

# -------------------------------
# Logging
# -------------------------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load("mobilenetv2_synthetic.pth", map_location=device))
model.to(device)
model.eval()
logging.info("Loaded MobileNetV2 model for inference.")

# -------------------------------
# Single-image prediction
# -------------------------------
def predict_image(image: Image.Image):
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
    label = "good" if pred.item() == 0 else "defective"
    return {"label": label, "confidence": float(confidence.item())}

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="MobileNetV2 Synthetic Defect API")
#app.mount("/static", StaticFiles(directory="."), name="static")
# Allow localhost and 127.0.0.1 for development
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
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
    return {"message": "MobileNetV2 Synthetic Defect Classifier API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = predict_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict_adv")
async def predict_adv(
    file: UploadFile = File(...),
    attack: str = Form("fgsm"),
    eps: Optional[float] = Form(None),
    eps_step: Optional[float] = Form(None),
    max_iter: Optional[int] = Form(None),
    targeted: Optional[bool] = Form(None),
    num_random_init: Optional[int] = Form(None),
    confidence: Optional[float] = Form(None),
    learning_rate: Optional[float] = Form(None),
    binary_search_steps: Optional[int] = Form(None),
    initial_const: Optional[float] = Form(None)
):
    try:
        logging.info(f"Received attack type: {attack}")
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Collect attack parameters dynamically
        attack_params = {}
        if eps is not None: attack_params["eps"] = eps
        if eps_step is not None: attack_params["eps_step"] = eps_step
        if max_iter is not None: attack_params["max_iter"] = max_iter
        if targeted is not None: attack_params["targeted"] = targeted
        if num_random_init is not None: attack_params["num_random_init"] = num_random_init
        if confidence is not None: attack_params["confidence"] = confidence
        if learning_rate is not None: attack_params["learning_rate"] = learning_rate
        if binary_search_steps is not None: attack_params["binary_search_steps"] = binary_search_steps
        if initial_const is not None: attack_params["initial_const"] = initial_const


        logging.info("calling generate_adversarial() from demo_attack") 
        result = generate_adversarial(
            image, model, device, attack_name=attack, attack_params=attack_params
        )

        adv_image = result["adversarial_image"]
        buffer = io.BytesIO()
        adv_image.save(buffer, format="PNG")
        adv_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = {
            "original_label": result["original"]["label"],
            "original_confidence": result["original"]["confidence"],
            "adv_label": result["adversarial"]["label"],
            "adv_confidence": result["adversarial"]["confidence"],
            "adv_image_base64": adv_base64
        }
        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"Adversarial prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/demo_ui.html")
def serve_ui():
        path = os.path.join(os.path.dirname(__file__), "demo_ui.html")
        return FileResponse(path)
