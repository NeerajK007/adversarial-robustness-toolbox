# models/mobilenet.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def load_mobilenet(num_classes, device=None, freeze_backbone=True):
    """
    Load MobileNetV2 pretrained on ImageNet and adapt for given number of classes.

    Args:
        num_classes (int): Number of output classes.
        device (torch.device, optional): 'cuda' or 'cpu'. Auto-detect if None.
        freeze_backbone (bool): Whether to freeze pretrained backbone layers.

    Returns:
        nn.Module: MobileNetV2 model ready for training/inference.
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model.to(device)


def save_model(model, path="mobilenetv2.pth"):
    """Save model weights to file."""
    torch.save(model.state_dict(), path)


def load_model(path, num_classes, device=None, freeze_backbone=True):
    """Load model weights from file into MobileNetV2."""
    model = load_mobilenet(num_classes, device, freeze_backbone)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
