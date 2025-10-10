
# train_model.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import json
import sys

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.classification.pytorch import PyTorchClassifier

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


def train_model_adversarial(model, train_loader, val_loader, device, epochs=5, lr=1e-3, attack=None):
    if attack is None:
        raise ValueError("You must provide an ART attack instance (e.g., FastGradientMethod, PGD).")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Clean loss
            outputs = model(images)
            loss_clean = criterion(outputs, labels)

            # Adversarial loss
            adv_images_np = attack.generate(images.cpu().numpy())  # ART expects numpy
            adv_images = torch.tensor(adv_images_np).to(device)
            outputs_adv = model(adv_images)
            loss_adv = criterion(outputs_adv, labels)

            # Combine
            loss = (loss_clean + loss_adv) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        metrics = evaluate_model(model, val_loader, device)
        history["train_loss"].append(running_loss / len(train_loader.dataset))
        history["val_accuracy"].append(metrics["accuracy"])
        logger.info(f"[Adversarial Epoch {epoch+1}/{epochs}] Loss={loss:.4f}, Val Acc={metrics['accuracy']:.4f}")

    return model, history



# -------------------------------
# Main
# -------------------------------
import os
def main():
    use_adversarial_training=False
    dataset_name = "mvtec"
    # -------------------------------
    # Project paths
    # -------------------------------
    project_root = os.path.dirname(__file__)
    logger.info("*** Start-point ***")

    # -------------------------------
    # Configuration (update here for each run)
    # -------------------------------
    class Config:
        dataset_name = "mvtec"          # Options: "mvtec", "traffic_signs", etc.
        model_name = "mobilenetv2"      # Options: "mobilenetv2", "resnet18", etc.
        epochs = 5
        batch_size = 16
        lr = 1e-3
        num_classes = 2    # Update based on dataset (e.g., 2 for mvtec binary, 4 for traffic signs)

    cfg = Config()
    # create timestamp like 20250911_121530
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Dynamically set data and save paths
    data_dir
    if cfg.dataset_name == "mvtec":
        data_dir = os.path.join(project_root, "data", "mvtec_anomaly_detection")
    else:
        data_dir = os.path.join(project_root, "data", "1_Indian_traffic_sign_classification_dataset_class4")
        
    if use_adversarial_training:
        save_path = os.path.join(project_root, "weights", f"{timestamp}_{cfg.model_name}_{cfg.dataset_name}_AdvTrained.pth")
    else:
        save_path = os.path.join(project_root, "weights", f"{timestamp}_{cfg.model_name}_{cfg.dataset_name}.pth")
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
    

    if use_adversarial_training:
        attack = ProjectedGradientDescent(
            estimator=get_art_classifier(model),
            eps=0.03,
            eps_step=0.01,
            max_iter=15,
            #targeted=targeted,
            #num_random_init=num_random_init,
            batch_size=16
        )
        model, history = train_model_adversarial(model, train_loader, val_loader, device,
                                                epochs=cfg.epochs, lr=cfg.lr, attack=attack)
    else:
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
