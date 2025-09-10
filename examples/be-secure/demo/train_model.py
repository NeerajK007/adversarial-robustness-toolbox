
# train_model.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import json
import sys

from .datasets.mvtec_loader import MVTecSyntheticDataset
from .datasets.traffic_sign_loader import TrafficSignDataset
from .models.mobilenet import load_mobilenet, save_model

# from datasets.traffic_sign_loader import TrafficSignDataset   # (later)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Logger setup
# -------------------------------
logger = logging.getLogger("TrainModel")
logger.setLevel(logging.INFO)

# If no handler is attached, add one
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  
    
# -------------------------------
# Evaluation helper
# -------------------------------
def evaluate_model(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# -------------------------------
# Training loop
# -------------------------------
def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = {"train_loss": [], "val_accuracy": []}
    logger.info(f"epochs= {epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate_model(model, val_loader, device)

        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(metrics["accuracy"])

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Val Acc: {metrics['accuracy']:.4f}"
        )

    return model, history


# -------------------------------
# Main
# -------------------------------
import os
def main():
    # -------------------------------
    # Project paths
    # -------------------------------
    project_root = os.path.dirname(os.path.dirname(__file__))
    logger.info("*** Start-point ***")

    # -------------------------------
    # Configuration (update here for each run)
    # -------------------------------
    class Config:
        dataset_name = "traffic_signs"          # Options: "mvtec", "traffic_signs", etc.
        model_name = "mobilenetv2"      # Options: "mobilenetv2", "resnet18", etc.
        epochs = 2
        batch_size = 16
        lr = 1e-3
        num_classes = 4    # Update based on dataset (e.g., 2 for mvtec binary, 4 for traffic signs)

    cfg = Config()
    
    # Dynamically set data and save paths
    #data_dir = os.path.join(project_root, "data", "mvtec_anomaly_detection")
    data_dir = os.path.join(project_root, "data", "Indian_traffic_sign_classification_dataset_class4")
    save_path = os.path.join(project_root, "demo", "weights", f"{cfg.model_name}_{cfg.dataset_name}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info(f"Dataset path: {data_dir}")
    logger.info(f"Model save path: {save_path}")
    logger.info(f"Training {cfg.model_name} on {cfg.dataset_name} dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Dataset selection
    # -------------------------------
    if cfg.dataset_name == "mvtec":
        train_dataset = MVTecSyntheticDataset(data_dir, is_train=True)
        val_dataset   = MVTecSyntheticDataset(data_dir, is_train=False, defect_prob=0.0)
    elif cfg.dataset_name == "traffic_signs":
        train_dataset = TrafficSignDataset(os.path.join(data_dir, "train"), is_train=True)
        val_dataset   = TrafficSignDataset(os.path.join(data_dir, "test"), is_train=False)
    else:
        raise ValueError(f"Dataset {cfg.dataset_name} not supported!")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # -------------------------------
    # Model selection
    # -------------------------------
    if cfg.model_name == "mobilenetv2":
        model = load_mobilenet(num_classes=cfg.num_classes, device=device, freeze_backbone=True)
    # elif cfg.model_name == "resnet18":
    #     model = load_resnet18(num_classes=cfg.num_classes, device=device, freeze_backbone=True)
    else:
        raise ValueError(f"Model {cfg.model_name} not supported!")

    # -------------------------------
    # Train
    # -------------------------------
    model, history = train_model(model, train_loader, val_loader, device,
                                 epochs=cfg.epochs, lr=cfg.lr)

    # -------------------------------
    # Save
    # -------------------------------
    save_model(model, save_path)
    logger.info(f"Training finished. Model saved to {save_path}")
    logger.info(f"History: {json.dumps(history, indent=2)}")


if __name__ == "__main__":
    main()
