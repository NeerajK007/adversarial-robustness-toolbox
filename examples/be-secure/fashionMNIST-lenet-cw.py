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
def load_fashion_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def build_classifier(model, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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


def run_carlini_wagner_attack(classifier, X, y, max_iter=10):
    attack = CarliniL2Method(
        classifier=classifier,
        max_iter=max_iter,
        batch_size=X.shape[0],
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
        train_loader, test_loader = load_fashion_mnist()

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
        X_adv = run_carlini_wagner_attack(classifier, X, y, max_iter=5)
        acc_adv = evaluate_accuracy(classifier, [(torch.tensor(X_adv), torch.tensor(y))], device, desc="after C&W attack")

        # 7. Report drop
        logging.info(f"Accuracy drop: {acc_clean:.4f} -> {acc_adv:.4f}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
