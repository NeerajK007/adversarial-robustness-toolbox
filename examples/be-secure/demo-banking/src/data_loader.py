"""
data_loader.py
---------------
Handles dataset loading, preprocessing, and splitting.

- Reads dataset configuration from YAML
- Loads open-source or synthetic data
- Performs scaling and train/test split
- Persists processed arrays for reproducibility
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.helpers import setup_logger, load_config


class DataLoader:
    def __init__(self, config_path: str):
        """
        Initialize DataLoader with configuration path.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config(config_path)
        self.scaler = StandardScaler()
        # output processed directory
        self.processed_dir = os.path.join(os.path.dirname(config_path), "..", "data", "processed")
        self.processed_dir = os.path.normpath(self.processed_dir)

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads dataset from CSV path defined in config.
        Returns raw DataFrame.
        """
        try:
            data_path = self.config.get("dataset", {}).get("path")
            if not data_path or not os.path.exists(data_path):
                self.logger.error(f"Dataset file not found: {data_path}")
                return None

            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded dataset with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.exception(f"Error loading dataset: {e}")
            return None

    def preprocess(self, df: pd.DataFrame):
        """
        Applies preprocessing: separates features/labels, scales, splits,
        persists processed arrays, and returns train/test splits as numpy arrays.

        Returns:
            X_train (np.ndarray), X_test (np.ndarray), y_train (np.ndarray), y_test (np.ndarray)
        """
        try:
            target_col = self.config.get("dataset", {}).get("target_column", "Class")
            test_size = self.config.get("dataset", {}).get("test_size", 0.2)
            random_state = self.config.get("dataset", {}).get("random_state", 42)

            if target_col not in df.columns:
                self.logger.error("Target column '%s' not found in dataframe columns.", target_col)
                return None, None, None, None

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Fit scaler on training features and transform both sets
            X_train_scaled = self.scaler.fit_transform(X_train_df.values)
            X_test_scaled = self.scaler.transform(X_test_df.values)

            # Convert labels to numpy arrays
            y_train = y_train_s.values
            y_test = y_test_s.values

            # Persist processed arrays for reproducibility and downstream use
            os.makedirs(self.processed_dir, exist_ok=True)
            np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train_scaled)
            np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test_scaled)
            np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
            np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

            self.logger.info(
                "Split complete. Train: %s, Test: %s. Processed arrays saved to %s",
                X_train_scaled.shape, X_test_scaled.shape, self.processed_dir
            )

            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            self.logger.exception(f"Error during preprocessing: {e}")
            return None, None, None, None
