## CRS
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import CarliniL2Method

import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from art.estimators.classification import PyTorchClassifier
import logging

logging.basicConfig(level=logging.INFO)
from datetime import datetime


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
# C&W (Carlini & Wagner) Attack
# -------------------------------
def cw_attack(
    image: Image.Image,
    model,
    device,
    confidence: float = 0.0,
    targeted: bool = False,
    learning_rate: float = 0.01,
    binary_search_steps: int = 10,
    max_iter: int = 10,
    initial_const: float = 0.01
):
    """
    Generate adversarial example using Carlini & Wagner L2 attack.
    Returns original & adversarial predictions + adversarial image.
    """
    start_time = datetime.now()
    logging.info(f"[CW_ATTACK] Start: {start_time.isoformat()}")
    logging.info(f"targeted- {targeted}")


    classifier = get_art_classifier(model)
    img_tensor = pil_to_tensor(image, device)

    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        #targeted=targeted,
        learning_rate=learning_rate,
        binary_search_steps=binary_search_steps,
        max_iter=max_iter,
        initial_const=initial_const,
        batch_size=1
    )
    adv_tensor = torch.tensor(attack.generate(img_tensor.cpu().numpy()))
    adv_img = tensor_to_pil(adv_tensor)

    # def predict(tensor):
    #     with torch.no_grad():
    #         outputs = model(tensor)
    #         probs = torch.softmax(outputs, dim=1)
    #         confidence_val, pred = torch.max(probs, dim=1)
    #         label = "good" if pred.item() == 0 else "defective"
    #         return label, float(confidence_val.item())

    # orig_label, orig_conf = predict(img_tensor)
    # adv_label, adv_conf = predict(pil_to_tensor(adv_img, device=device))

    end_time = datetime.now()
    logging.info(f"[CW_ATTACK] End: {end_time.isoformat()} (Duration: {end_time - start_time})")

    return {
        # "original": {"label": orig_label, "confidence": orig_conf},
        # "adversarial": {"label": adv_label, "confidence": adv_conf},
        "adversarial_image": adv_img
    }
    
# -------------------------------
# PGD(Projected Gradient Descent) Attack
# -------------------------------
def pgd_attack(
    image: Image.Image,
    model,
    device,
    eps: float = 0.1,
    eps_step: float = 0.01,
    max_iter: int = 10,
    targeted: bool = False,
    num_random_init: int = 0
):
    """
    Generate adversarial example using Projected Gradient Descent (PGD) attack.
    Returns original & adversarial predictions + adversarial image.
    """
    start_time = datetime.now()
    logging.info(f"[PGD_ATTACK] Start: {start_time.isoformat()}")
    logging.info(f"targeted- {targeted}")

    classifier = get_art_classifier(model)
    img_tensor = pil_to_tensor(image, device)
    
    #targeted = "y" if targeted else "n"

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=targeted,
        num_random_init=num_random_init,
        batch_size=1
    )
    adv_tensor = torch.tensor(attack.generate(img_tensor.cpu().numpy()))
    adv_img = tensor_to_pil(adv_tensor)

    # def predict(tensor):
    #     with torch.no_grad():
    #         outputs = model(tensor)
    #         probs = torch.softmax(outputs, dim=1)
    #         confidence_val, pred = torch.max(probs, dim=1)
    #         label = "good" if pred.item() == 0 else "defective"
    #         return label, float(confidence_val.item())

    # orig_label, orig_conf = predict(img_tensor)
    # adv_label, adv_conf = predict(pil_to_tensor(adv_img, device=device))

    end_time = datetime.now()
    logging.info(f"[PGD_ATTACK] End: {end_time.isoformat()} (Duration: {end_time - start_time})")

    return {
        # "original": {"label": orig_label, "confidence": orig_conf},
        # "adversarial": {"label": adv_label, "confidence": adv_conf},
        "adversarial_image": adv_img
    }

# -------------------------------
# FGSM (Fast Gradient Sign Method) Attack
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
    # def predict(tensor):
    #     with torch.no_grad():
    #         outputs = model(tensor)
    #         probs = torch.softmax(outputs, dim=1)
    #         confidence, pred = torch.max(probs, dim=1)
    #         label = "good" if pred.item() == 0 else "defective"
    #         return label, float(confidence.item())

    #orig_label, orig_conf = predict(img_tensor)
    #adv_label, adv_conf = predict(pil_to_tensor(adv_img, device=device))

    return {
        # "original": {"label": orig_label, "confidence": orig_conf},
        # "adversarial": {"label": adv_label, "confidence": adv_conf},
        "adversarial_image": adv_img  # <-- keep as PIL.Image
}

# -------------------------------
# Future Attacks Placeholder
# -------------------------------
def generate_adversarial(image: Image.Image, model, device, attack_name="fgsm", attack_params=None):
   
    logging.info("in-generate_adversarial - ", datetime.now().isoformat())
    if attack_params is None:
        attack_params = {}

    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        logging.info("attack_name == fgsm")
        eps = attack_params.get("eps", 0.1)
        return fgsm_attack(image, model=model, device=device, eps=eps)
    elif attack_name == "pgd":
        logging.info("attack_name == pgd")
        targeted_str = attack_params.get("targeted", "false")
        targeted = str(targeted_str).lower() == "true"
        return pgd_attack(
            image,
            model=model,
            device=device,
            eps=attack_params.get("eps", 0.1),
            eps_step=attack_params.get("eps_step", 0.01),
            max_iter=attack_params.get("max_iter", 10),
            targeted=targeted,
            num_random_init=attack_params.get("num_random_init", 0)
        )
    elif attack_name == "cw":
        logging.info("attack_name == cw")
        return cw_attack(
            image,
            model=model,
            device=device,
            confidence=attack_params.get("confidence", 0.0),
            targeted=attack_params.get("targeted", False),
            learning_rate=attack_params.get("learning_rate", 0.01),
            binary_search_steps=attack_params.get("binary_search_steps", 10),
            max_iter=attack_params.get("max_iter", 10),
            initial_const=attack_params.get("initial_const", 0.01)
    )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
