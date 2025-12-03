"""
surrogate.py
------------
Lightweight PyTorch MLP surrogate trainer used to craft gradient-based
adversarial examples for transfer attacks.

Provides:
 - SurrogateTrainer: build, train, and inference helpers.
 - Methods are intentionally small and well-logged.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.utils.helpers import setup_logger

LOG = setup_logger(__name__)


class SurrogateTrainer:
    """
    Train a compact MLP surrogate on preprocessed (scaled) numpy arrays.

    Usage:
      trainer = SurrogateTrainer(input_dim=30, hidden_dim=64, num_classes=2, device='cpu')
      trainer.fit(X_train, y_train, epochs=10, batch_size=256)
      probs = trainer.predict_proba(X_test)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._build_mlp(input_dim, hidden_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        LOG.info("Initialized surrogate MLP (input=%s, hidden=%s, classes=%s, device=%s)",
                 input_dim, hidden_dim, num_classes, self.device)

    def _build_mlp(self, input_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        class MLP(nn.Module):
            def __init__(self, in_dim, hid, out_d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid),
                    nn.ReLU(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Linear(hid, out_d)
                )

            def forward(self, x):
                return self.net(x)

        return MLP(input_dim, hidden_dim, out_dim)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 256,
            lr: float = 1e-3):
        """
        Train surrogate on numpy arrays (already scaled). Logs per-epoch loss.

        Args:
            X_train: shape (n_samples, n_features)
            y_train: shape (n_samples,)
            epochs: number of training epochs
            batch_size: batch size
            lr: learning rate
        """
        try:
            self.model.train()
            X_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(y_train.astype(np.int64)).to(self.device)
            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            start = time.time()
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(dataset)
                LOG.info("Surrogate epoch %d/%d - loss=%.6f", epoch, epochs, epoch_loss)
            elapsed = time.time() - start
            LOG.info("Surrogate training completed in %.2fs", elapsed)
            return self.model
        except Exception as exc:
            LOG.exception("Surrogate training failed: %s", exc)
            raise

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Return softmax probabilities for input numpy array X.

        Args:
            X: numpy array [n_samples, n_features]
            batch_size: inference batch size

        Returns:
            probs: numpy array [n_samples, num_classes]
        """
        try:
            self.model.eval()
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            with torch.no_grad():
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
        except Exception as exc:
            LOG.exception("Surrogate predict_proba failed: %s", exc)
            raise
