"""
evaluator.py
------------
Evaluate models on clean (non-adversarial) data.
Provides:
 - evaluate_classifier(model, X_test, y_test)
 - save_metrics(metrics, out_path)

Logs Info/Error for each step.
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from src.utils.helpers import setup_logger


logger = setup_logger(__name__)


def evaluate_classifier(model, X_test, y_test, average="binary"):
    """
    Evaluate a scikit-learn style classifier on test data.

    Args:
        model: trained model with predict(X) method
        X_test: array-like, test features
        y_test: array-like, true labels
        average: str, averaging method for multi-class precision/recall/f1

    Returns:
        dict: metrics {accuracy, precision, recall, f1, confusion_matrix}
    """
    try:
        logger.info("Starting clean evaluation.")
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average=average, zero_division=0))
        rec = float(recall_score(y_test, preds, average=average, zero_division=0))
        f1 = float(f1_score(y_test, preds, average=average, zero_division=0))
        cm = confusion_matrix(y_test, preds).tolist()

        metrics = {
            "clean_accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm
        }

        logger.info(
            "Evaluation complete. Accuracy: %.4f, Recall: %.4f, Precision: %.4f",
            acc, rec, prec
        )
        return metrics

    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        return {}


def save_metrics(metrics: dict, out_path: str):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: dict of evaluation metrics
        out_path: destination file path
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {out_path}")
    except Exception as e:
        logger.exception(f"Failed to save metrics to {out_path}: {e}")
