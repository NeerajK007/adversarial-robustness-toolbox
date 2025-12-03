# datasets/traffic_sign_loader.py
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import logging
import sys

# Logger setup
logger = logging.getLogger("TrafficSignLoader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, classes=None, img_size=224, is_train=True):
        """
        Args:
            root_dir (str): Path to traffic sign dataset.
            classes (list): List of class folder names to include.
            img_size (int): Resize image to this size.
            is_train (bool): Load train or test data.
        """
        self.samples = []
        self.img_size = img_size
        self.is_train = is_train

        if classes is None:
            classes = sorted(os.listdir(root_dir))

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        logger.info(f"Classes: {self.class_to_idx}")
        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            #logger.info(f"Looking for images in {cls_path}")

            exts = ["*.png", "*.jpg", "*.jpeg"]
            for ext in exts:
                self.samples.extend([(p, self.class_to_idx[cls_name]) 
                                     for p in glob.glob(os.path.join(cls_path, ext))])

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Loaded {len(self.samples)} samples for {'train' if is_train else 'test'}")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

