#!/usr/bin/env python3
"""
model_loader.py
Unified model loading logic shared across demo API and robustness evaluation.
Delegates to architecture-specific load_model() in the models directory.
"""

import logging
import torch
from model_registry import get_model_cfg
from models.mobilenet import load_model as load_mobilenet_model  #main import

_MODEL_CACHE = {}
logger = logging.getLogger("ART-ModelLoader")


def load_demo_model(demo_type: str, device, variant: str = "normal", reload: bool = False):
    """
    Load and return a demo model for the given demo_type.

    Args:
        demo_type (str): Key from MODEL_REGISTRY
        device (torch.device): Target device (cpu/cuda)
        variant (str): 'normal' or 'adv_trained'
        reload (bool): Force reload even if cached

    Returns:
        torch.nn.Module
    """
    cache_key = f"{demo_type}_{variant}_{device}"
    if not reload and cache_key in _MODEL_CACHE:
        logger.info(f"Reusing cached model for {demo_type} ({variant})")
        return _MODEL_CACHE[cache_key]

    cfg = get_model_cfg(demo_type)
    weights_path = cfg.get("weights_path")
    if variant == "adv_trained" and cfg.get("adv_weights_path"):
        weights_path = cfg["adv_weights_path"]
        logger.info(f"Using adversarially trained weights: {weights_path}")

    num_classes = cfg.get("num_classes", 2)

    # Clean delegation — use load_model() directly
    model = load_mobilenet_model(
        path=weights_path,
        num_classes=num_classes,
        device=device,
        freeze_backbone=True
    )

    #model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model

    logger.info(f"Model loaded and cached for demo_type={demo_type}")
    return model


def clear_model_cache():
    """Clear in-memory cached models."""
    _MODEL_CACHE.clear()
    logger.info("Cleared model cache.")
