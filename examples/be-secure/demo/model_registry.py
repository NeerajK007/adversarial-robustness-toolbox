#!/usr/bin/env python3
"""
model_registry.py
Central registry for demo model configurations.
This isolates model metadata (architecture, weights, and classes)
so both demo_api.py and evaluate_robustness.py can reuse it.
"""

from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

MODEL_REGISTRY = {
    "Tile-defect-misclassification": {
        "model_fn": models.mobilenet_v2,
        "weights_enum": MobileNet_V2_Weights.IMAGENET1K_V1,
        "num_classes": 2,
        "weights_path": "weights/20250925_110902_mobilenetv2_mvtec.pth",
        "adv_weights_path": "weights/20250925_110902_mobilenetv2_mvtec_AdvTrained.pth",
        "pretrained": "ImageNet",
        "dataset_name": "mvtec_anomaly_detection",
        "variant_supported": ["normal"],
        "variant":"normal",
        "demo_type":"Tile-defect-misclassification"
    },
    "Indian-trafic-signal-misclassification": {
        "model_fn": models.mobilenet_v2,
        "weights_enum": MobileNet_V2_Weights.IMAGENET1K_V1,
        "num_classes": 4,
        "weights_path": "weights/20250919_133106_mobilenetv2_traffic_signs.pth",
        "adv_weights_path": "weights/20250918_154508_mobilenetv2_traffic_signs_AdvTrained.pth",
        "pretrained": "ImageNet",
        "dataset_name": "Indian_traffic_sign_classification_dataset",
        "variant_supported": ["normal", "adv_trained"],
        "variant":"normal",
        "demo_type":"Indian-trafic-signal-misclassification"
    }
}


def get_model_cfg(demo_type: str) -> dict:
    """
    Retrieve configuration for a specific demo type.

    Args:
        demo_type (str): Key from MODEL_REGISTRY

    Returns:
        dict: model configuration details

    Raises:
        KeyError: if demo_type is not registered
    """
    if demo_type not in MODEL_REGISTRY:
        raise KeyError(f"Model config not found for demo_type: {demo_type}")
    return MODEL_REGISTRY[demo_type]
