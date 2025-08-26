import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import logging

logging.basicConfig(level=logging.INFO)

# -------------------------------
# ART Classifier Wrapper
# -------------------------------
def get_art_classifier(model):
    """
    Wrap PyTorch model as ART classifier for evasion attacks.
    """
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=None,  # Not needed for inference
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(0, 1),
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    return classifier

# -------------------------------
# Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Utilities
# -------------------------------
def pil_to_tensor(image: Image.Image, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor

def tensor_to_pil(tensor: torch.Tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def pil_to_base64(img: Image.Image):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -------------------------------
# FGSM Attack
# -------------------------------
def fgsm_attack(image: Image.Image, model, device, eps: float = 0.1):
    """
    Generate adversarial example using FGSM.
    Returns original & adversarial predictions + adversarial image.
    """
    classifier = get_art_classifier(model)
    img_tensor = pil_to_tensor(image, device)

    # Generate adversarial tensor
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    adv_tensor = torch.tensor(attack.generate(img_tensor.cpu().numpy()))
    adv_img = tensor_to_pil(adv_tensor)

    # Prediction helper
    def predict(tensor):
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            label = "good" if pred.item() == 0 else "defective"
            return label, float(confidence.item())

    orig_label, orig_conf = predict(img_tensor)
    adv_label, adv_conf = predict(pil_to_tensor(adv_img, device=device))

    return {
        "original": {"label": orig_label, "confidence": orig_conf},
        "adversarial": {"label": adv_label, "confidence": adv_conf},
        "adversarial_image": adv_img  # <-- keep as PIL.Image
}

# -------------------------------
# Future Attacks Placeholder
# -------------------------------
def generate_adversarial(image: Image.Image, model, device, attack_name="fgsm", attack_params=None):
    if attack_params is None:
        attack_params = {}

    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        eps = attack_params.get("eps", 0.1)
        return fgsm_attack(image, model=model, device=device, eps=eps)
    elif attack_name == "pgd":
        raise NotImplementedError("PGD attack not implemented yet")
    elif attack_name == "cw":
        raise NotImplementedError("C&W attack not implemented yet")
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
