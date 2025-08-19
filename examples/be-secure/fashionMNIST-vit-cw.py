import logging
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTForImageClassification
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------
# Load Dataset
# ----------------------
def load_fashion_mnist():
    try:
        logging.info("Loading Fashion-MNIST dataset from HuggingFace...")
        dataset = load_dataset("fashion_mnist")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load Fashion-MNIST: {e}")
        raise

# ----------------------
# Preprocessing
# ----------------------
def preprocess_data(dataset, split="test", num_samples=30):
    try:
        logging.info(f"Preprocessing {num_samples} samples from the {split} set...")
        images = dataset[split]["image"][:num_samples]
        labels = np.array(dataset[split]["label"][:num_samples])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # Converts to [0,1] range
        ])
        processed_images = [transform(img) for img in images]
        X = torch.stack(processed_images).numpy()
        return X, labels
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

# ----------------------
# Model Wrapper
# ----------------------
class HFViTWrapper(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, x):
        # x: (N,3,224,224) tensor in [0,1]
        # Hugging Face expects normalized pixel_values
        pixel_values = self.processor(images=x, return_tensors="pt").pixel_values.to(x.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

# ----------------------
# Load Hugging Face Model
# ----------------------
def load_hf_model(model_name="WinKawaks/vit-tiny-patch16-224", num_classes=10, device="cpu"):
    try:
        logging.info(f"Loading HuggingFace model: {model_name}")
        # Load processor (AutoImageProcessor works for ViT family)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        model = HFViTWrapper(model, processor).to(device)
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load HuggingFace model: {e}")
        raise

# ----------------------
# ART Classifier Wrapper
# ----------------------
def wrap_with_art_classifier(model, device, num_classes=10):
    try:
        logging.info("Wrapping model with ART PyTorchClassifier...")
        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=num_classes,
            clip_values=(0, 1),  # since preprocessing gave [0,1]
            device_type="gpu" if torch.cuda.is_available() else "cpu"
        )
        return classifier
    except Exception as e:
        logging.error(f"Failed to wrap model with ART: {e}")
        raise

# ----------------------
# Evaluation
# ----------------------
def evaluate_accuracy(classifier, X, y, desc=""):
    try:
        logging.info(f"Evaluating accuracy {desc}...")
        preds = np.argmax(classifier.predict(X), axis=1)
        acc = np.mean(preds == y)
        logging.info(f"Accuracy {desc}: {acc:.4f}")
        return acc
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# ----------------------
# Carlini & Wagner Attack
# ----------------------
def run_carlini_wagner_attack(classifier, X, y, max_iter=3):
    try:
        logging.info("Running Carlini & Wagner L2 attack...")
        attack = CarliniL2Method(
            classifier=classifier,
            max_iter=max_iter,
            learning_rate=0.01,
            batch_size=10
        )
        X_adv = attack.generate(x=X)
        return X_adv
    except Exception as e:
        logging.error(f"Error during Carlini & Wagner attack: {e}")
        raise

# ----------------------
# Main Pipeline
# ----------------------
def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = load_fashion_mnist()
        model, processor = load_hf_model(device=device)
        X, y = preprocess_data(dataset, split="test", num_samples=10)

        classifier = wrap_with_art_classifier(model, device)
        acc_clean = evaluate_accuracy(classifier, X, y, desc="on clean data")

        X_adv = run_carlini_wagner_attack(classifier, X, y, max_iter=3)
        acc_adv = evaluate_accuracy(classifier, X_adv, y, desc="on adversarial data (C&W)")

        logging.info(f"Accuracy drop after C&W attack: {acc_clean:.4f} -> {acc_adv:.4f}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
