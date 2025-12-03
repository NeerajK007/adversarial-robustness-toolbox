#!/usr/bin/env python3
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------
# 1) Model: Simple CNN for MNIST
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # [B,32,28,28]
        x = self.pool(x)                 # [B,32,14,14]
        x = self.relu(self.conv2(x))     # [B,64,14,14]
        x = self.pool(x)                 # [B,64,7,7]
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)                  # raw logits
        return x


# ---------------------------
# 2) Data: MNIST -> NumPy for ART
# ---------------------------
def load_mnist_numpy(limit_train=1000, limit_test=100):
    """
    Loads MNIST and returns NumPy arrays suitable for ART:
      x_* : float32 in [0,1], shape (N,1,28,28)
      y_* : int64 class indices shape (N,)
      y_*_oh : float32 one-hot shape (N,10)
    """
    logging.info("Loading MNIST (NumPy arrays for ART)...")
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    # Use raw tensors (faster than iterating loaders), then normalize
    x_train = train.data.numpy().astype(np.float32) / 255.0   # (60000,28,28)
    y_train = train.targets.numpy().astype(np.int64)          # (60000,)
    x_test = test.data.numpy().astype(np.float32) / 255.0     # (10000,28,28)
    y_test = test.targets.numpy().astype(np.int64)            # (10000,)

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=1)                 # (N,1,28,28)
    x_test = np.expand_dims(x_test, axis=1)                   # (N,1,28,28)

    # Subset for quick CPU runs
    if limit_train is not None:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test is not None:
        x_test = x_test[:limit_test]
        y_test = y_test[:limit_test]

    # One-hot for ART fit()
    nb_classes = 10
    y_train_oh = np.eye(nb_classes, dtype=np.float32)[y_train]  # (N,10)

    logging.info(f"x_train: {x_train.shape}, y_train: {y_train.shape} "
                 f"| x_test: {x_test.shape}, y_test: {y_test.shape}")
    return x_train, y_train, y_train_oh, x_test, y_test


# ---------------------------
# 3) ART Classifier
# ---------------------------
def build_art_classifier(model, device, lr=1e-3):
    """
    Wraps the PyTorch model in an ART PyTorchClassifier.
    """
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )
    return classifier


# ---------------------------
# 4) Train via ART
# ---------------------------
def train_with_art(classifier, x_train, y_train_oh, batch_size=16, epochs=2):
    logging.info(f"Training via ART: epochs={epochs}, batch_size={batch_size}")
    classifier.fit(x_train, y_train_oh, batch_size=batch_size, nb_epochs=epochs)
    logging.info("Training complete.")


# ---------------------------
# 5) Evaluate helper
# ---------------------------
def accuracy_from_predictions(preds: np.ndarray, y_true_idx: np.ndarray) -> float:
    """
    preds: np.ndarray (N, nb_classes) from classifier.predict()
    y_true_idx: np.ndarray (N,) int labels
    """
    y_pred_idx = np.argmax(preds, axis=1)
    return float(np.mean(y_pred_idx == y_true_idx)) * 100.0


def evaluate_clean(classifier, x_test, y_test_idx):
    logging.info("Evaluating on clean test set...")
    preds = classifier.predict(x_test)
    acc = accuracy_from_predictions(preds, y_test_idx)
    logging.info(f"Clean accuracy: {acc:.2f}%")
    return acc


# ---------------------------
# 6) FGSM Attack via ART
# ---------------------------
def run_fgsm(classifier, x, eps=0.1, batch_size=16):
    logging.info(f"Generating FGSM adversarial examples: eps={eps}")
    attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=batch_size)
    x_adv = attack.generate(x=x)
    logging.info("FGSM generation complete.")
    return x_adv


def evaluate_adversarial(classifier, x_adv, y_test_idx, tag="FGSM"):
    logging.info(f"Evaluating on adversarial ({tag}) test set...")
    preds_adv = classifier.predict(x_adv)
    acc_adv = accuracy_from_predictions(preds_adv, y_test_idx)
    logging.info(f"Adversarial ({tag}) accuracy: {acc_adv:.2f}%")
    return acc_adv


# ---------------------------
# 7) Main Pipeline
# ---------------------------
def main():
    # Repro & device
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Hyperparameters for quick CPU runs
    EPOCHS = 2
    BATCH_SIZE = 16
    LR = 1e-3
    LIMIT_TRAIN = 1000
    LIMIT_TEST = 100
    EPSILONS = [0.05, 0.1, 0.2]

    # Load data
    x_train, y_train_idx, y_train_oh, x_test, y_test_idx = load_mnist_numpy(
        limit_train=LIMIT_TRAIN, limit_test=LIMIT_TEST
    )

    # Build model + ART classifier
    model = CNN().to(device)
    classifier = build_art_classifier(model, device, lr=LR)

    # Train
    train_with_art(classifier, x_train, y_train_oh, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Clean evaluation
    acc_clean = evaluate_clean(classifier, x_test, y_test_idx)

    # FGSM sweeps
    results = []
    for eps in EPSILONS:
        x_test_adv = run_fgsm(classifier, x_test, eps=eps, batch_size=BATCH_SIZE)
        acc_adv = evaluate_adversarial(classifier, x_test_adv, y_test_idx, tag=f"FGSM ε={eps}")
        results.append({"eps": eps, "accuracy": round(acc_adv, 3)})

    # Report
    report = {
        "REPORT_TYPE": "adversarial_attack",
        "DATASET": "MNIST",
        "MODEL": "Custom-CNN",
        "ATTACK_METHOD": "FGSM",
        "TRAINING": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
          },
        "MODEL_BEFORE": {
                "accuracy": round(acc_clean, 3)
          },
         "MODEL_AFTER": {
            "PARAMETERS": {"epsilons": EPSILONS},
            "RESULTS": results,
          },
        "NOTES": "ART pipeline: trained via PyTorchClassifier; evaluated clean and under FGSM at multiple epsilons.",
    }

    logging.info("Final Report:\n" + json.dumps(report, indent=2))
    #print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
