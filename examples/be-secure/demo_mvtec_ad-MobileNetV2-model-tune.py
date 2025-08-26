import os
import glob
import logging
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# Dataset with synthetic defects
# -------------------------------
class MVTecSyntheticDataset(Dataset):
    def __init__(self, root_dir, categories=None, img_size=224, is_train=True, defect_prob=0.5):
        self.samples = []
        self.is_train = is_train
        self.defect_prob = defect_prob
        self.img_size = img_size

        if categories is None:
            categories = os.listdir(root_dir)

        exts = ["*.png", "*.jpg", "*.jpeg"]

        for cat in categories:
            cat_path = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_path):
                continue
            train_or_test = "train" if is_train else "test"

            if is_train:
                good_dir = os.path.join(cat_path, "train", "good")
                for ext in exts:
                    self.samples.extend(glob.glob(os.path.join(good_dir, ext)))
            else:
                test_dir = os.path.join(cat_path, "test")
                for defect_type in os.listdir(test_dir):
                    label = 0 if defect_type == "good" else 1
                    defect_dir = os.path.join(test_dir, defect_type)
                    for ext in exts:
                        for img_path in glob.glob(os.path.join(defect_dir, ext)):
                            self.samples.append((img_path, label))

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def add_synthetic_defect(self, image):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        choice = random.choice(["occlusion", "line", "noise"])
        w, h = img.size
        if choice == "occlusion":
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0,0,0))
        elif choice == "line":
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            draw.line([x1, y1, x2, y2], fill=(255,255,255), width=3)
        elif choice == "noise":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(2,5)))
        return img

    def __getitem__(self, idx):
        if self.is_train:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            label = 1 if random.random() < self.defect_prob else 0
            if label == 1:
                image = self.add_synthetic_defect(image)
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")

        image = self.transform(image)
        return image, label

# -------------------------------
# Model loader
# -------------------------------
def load_model(device=None):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Loaded MobileNetV2 (last layer trainable) on {device}.")
    return model

# -------------------------------
# Evaluation
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
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

# -------------------------------
# Training
# -------------------------------
def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=lr)
    history = {"train_loss": [], "val_accuracy": []}

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
        logging.info(
            f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} "
            f"Val Acc: {metrics['accuracy']:.4f}"
        )
    return model, history

# -------------------------------
# Main
# -------------------------------
def main():
    print("✅ Starting MobileNetV2 Synthetic Defect Training")

    data_dir = "./data/mvtec_anomaly_detection"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MVTecSyntheticDataset(data_dir, is_train=True)
    test_dataset  = MVTecSyntheticDataset(data_dir, is_train=False, defect_prob=0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(device=device)
    model, history = train_model(model, train_loader, test_loader, device, epochs=2, lr=1e-3)

    # Save the trained model
    torch.save(model.state_dict(), "mobilenetv2_synthetic.pth")
    logging.info("Model saved as mobilenetv2_synthetic.pth")
    logging.info(f"Training history: {json.dumps(history, indent=2)}")

if __name__ == "__main__":
    main()
