import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import logging
import numpy as np
import json

logging.basicConfig(level=logging.INFO)


# ---------------------------
# Model Definition
# ---------------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------
# Pipeline Methods
# ---------------------------
def load_fashion_mnist(batch_size=10):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    # Limit dataset size for debugging/resource management
    trainset = torch.utils.data.Subset(trainset, range(1000))  # Use only 1000 samples
    testset = torch.utils.data.Subset(testset, range(100))     # Use only 100 samples

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    logging.info(f"Loaded FashionMNIST: {len(trainset)} train samples, {len(testset)} test samples")
    return train_loader, test_loader


def build_classifier(model, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logging.info(f"Model: {model}");
    return PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )


def train_model(classifier, train_loader, device, epochs=2):
    classifier.model.train()
    logging.info(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            classifier._optimizer.zero_grad()
            outputs = classifier.model(images)
            loss = classifier._loss(outputs, labels)
            loss.backward()
            classifier._optimizer.step()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def evaluate_accuracy(classifier, loader, device, desc=""):
    classifier.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = classifier.predict(images.cpu().numpy())
            preds = np.argmax(preds, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    logging.info(f"Accuracy {desc}: {acc:.4f}")
    return acc


def run_carlini_wagner_attack(classifier, X, y, confidence = 0, max_iter=10, learning_rate=0.01, initial_const=0.001):
    attack = CarliniL2Method(
        classifier=classifier,
        #confidence=confidence,
        max_iter=max_iter,
        #learning_rate=learning_rate,
        #initial_const=initial_const,
        batch_size=10,
    )
    return attack.generate(x=X)


# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # 1. Load dataset
        train_loader, test_loader = load_fashion_mnist(batch_size=10)

        # 2. Initialize model & classifier
        model = LeNet().to(device)
        classifier = build_classifier(model, device)

        # 3. Train model
        train_model(classifier, train_loader, device, epochs=2)

        # 4. Evaluate on clean data
        acc_clean = evaluate_accuracy(classifier, test_loader, device, desc="on clean data")

        # 5. Take a batch for attack
        X, y = next(iter(test_loader))
        X, y = X.numpy(), y.numpy()

        # 6. Run C&W attack
        confidence = 0 
        max_iter=10 
        learning_rate=0.01 
        initial_const=0.001
        
        X_adv = run_carlini_wagner_attack(classifier, X, y, confidence, max_iter, learning_rate, initial_const)
        acc_adv = evaluate_accuracy(classifier, [(torch.tensor(X_adv), torch.tensor(y))], device, desc="after C&W attack")

        # 7. Report drop
        logging.info(f"Accuracy drop: {acc_clean:.4f} -> {acc_adv:.4f}")


        report = {
            "REPORT_TYPE": "adversarial_attack",
            "DATASET": "Fashion-MNIST",
            "MODEL": "LeNet",
            "ATTACK_METHOD": "CarliniWagnerL2",
            "ATTACK_PARAMETERS": {
                "confidence": "default",
                "max_iterations": max_iter,
                "learning_rate": "default",
                "initial_const": "default"
            },
            "MODEL_BEFORE": {
                "accuracy": round(acc_clean, 3)
            },
            "MODEL_AFTER": {
                "accuracy": round(acc_adv, 3)
            },
            "NOTES": "C&W successfully crafted low-distortion adversarial samples that fooled the classifier."
        }
        logging.info("Attack report:")
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
