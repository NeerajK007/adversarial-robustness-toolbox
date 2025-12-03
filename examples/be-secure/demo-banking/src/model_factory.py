"""
model_factory.py
----------------
Provides a factory interface to create and load models
based on configuration.

Supports:
- Tree-based models (XGBoost, LightGBM, RandomForest)
- Neural network (MLP using PyTorch)
"""

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.helpers import setup_logger, load_config



class ModelFactory:
    def __init__(self, config_path: str):
        """
        Initializes ModelFactory with provided configuration.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config(config_path)
        self.model = None

    def create_model(self):
        """
        Creates a model instance based on configuration.
        """
        try:
            model_type = self.config.get("model", {}).get("type", "xgboost").lower()

            if model_type == "xgboost":
                params = self.config.get("model", {}).get("params", {})
                self.model = XGBClassifier(**params)
                self.logger.info("Initialized XGBoost model.")

            elif model_type == "randomforest":
                params = self.config.get("model", {}).get("params", {})
                self.model = RandomForestClassifier(**params)
                self.logger.info("Initialized RandomForest model.")

            elif model_type == "mlp":
                params = self.config.get("model", {}).get("params", {})
                input_dim = params.get("input_dim", 30)
                hidden_dim = params.get("hidden_dim", 64)
                output_dim = params.get("output_dim", 2)
                self.model = self._create_mlp(input_dim, hidden_dim, output_dim)
                self.logger.info("Initialized MLP model.")

            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None

            return self.model
        except Exception as e:
            self.logger.exception(f"Error creating model: {e}")
            return None

    def _create_mlp(self, input_dim, hidden_dim, output_dim):
        """
        Internal helper to create a simple feed-forward MLP.
        """
        class MLP(nn.Module):
            def __init__(self, in_dim, hid_dim, out_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, out_dim)
                )

            def forward(self, x):
                return self.layers(x)

        model = MLP(input_dim, hidden_dim, output_dim)
        return model

    def save_model(self, save_path: str):
        """
        Saves trained model to the specified path.
        """
        try:
            if isinstance(self.model, (XGBClassifier, RandomForestClassifier)):
                joblib.dump(self.model, save_path)
            elif isinstance(self.model, nn.Module):
                torch.save(self.model.state_dict(), save_path)
            else:
                self.logger.error("No valid model to save.")
                return
            self.logger.info(f"Model saved successfully to {save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving model: {e}")

    def load_model(self, load_path: str, model_type: str = None):
        """
        Loads model from disk based on type.
        """
        try:
            if not os.path.exists(load_path):
                self.logger.error(f"Model file not found: {load_path}")
                return None

            if model_type == "mlp":
                self.model.load_state_dict(torch.load(load_path))
            else:
                self.model = joblib.load(load_path)

            self.logger.info(f"Model loaded successfully from {load_path}")
            return self.model
        except Exception as e:
            self.logger.exception(f"Error loading model: {e}")
            return None
