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
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from demo_fgsm_attack import generate_adversarial  # your attack module
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Logging
# -------------------------------
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

@app.post("/predict_adv/")
async def predict_adv(file: UploadFile = File(...), attack: str = "fgsm", eps: float = 0.1):
    """
    attack: 'fgsm', 'pgd', 'cw' (only FGSM implemented for now)
    eps: epsilon for FGSM
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #adv_result = generate_adversarial(image, attack_name=attack, attack_params={"eps": eps})

        adv_result = generate_adversarial(image, model=model, device=device,
                                  attack_name=attack, attack_params={"eps": eps})

        # adv_result should return dict: {"adv_image": PIL.Image, "pred_label": str, "confidence": float}
        adv_image = adv_result["adversarial_image"]
        buffer = io.BytesIO()
        adv_image.save(buffer, format="PNG")
        adv_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = {
            "original_label": adv_result["original"]["label"],
            "original_confidence": adv_result["original"]["confidence"],
            "adv_label": adv_result["adversarial"]["label"],
            "adv_confidence": adv_result["adversarial"]["confidence"],
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
