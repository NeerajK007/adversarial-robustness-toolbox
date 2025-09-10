# datasets/mvtec_loader.py
import os
import glob
import random
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

import logging
import sys

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
    

class MVTecSyntheticDataset(Dataset):
    def __init__(self, root_dir, categories=None, img_size=224, is_train=True, defect_prob=0.5):
        """
        Args:
            root_dir (str): Path to dataset root (mvtec_anomaly_detection).
            categories (list): List of categories (folders) to include.
            img_size (int): Resize size (default=224 for MobileNetV2).
            is_train (bool): Train mode or test mode.
            defect_prob (float): Probability of adding synthetic defect (train only).
        """
        logger.info(f"Initializing MVTecSyntheticDataset with root: {root_dir}, is_train: {is_train}, defect_prob: {defect_prob}")
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

            if is_train:
                logger.info(f"Loading training data for category: {cat}")
                good_dir = os.path.join(cat_path, "train", "good")
                for ext in exts:
                    self.samples.extend(glob.glob(os.path.join(good_dir, ext)))
            else:
                logger.info(f"Loading testing data for category: {cat}")
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
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def add_synthetic_defect(self, image):
        #logger.info("Adding synthetic defect to image")
        """Randomly apply occlusion, line, or blur to simulate defects."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        choice = random.choice(["occlusion", "line", "noise"])
        w, h = img.size
        if choice == "occlusion":
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        elif choice == "line":
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            draw.line([x1, y1, x2, y2], fill=(255, 255, 255), width=3)
        elif choice == "noise":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 5)))
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
        #logger.info("completing MVTecSyntheticDataset operation")
        return image, label
