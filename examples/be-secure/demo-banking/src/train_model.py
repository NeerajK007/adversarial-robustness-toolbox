"""
train_model.py
---------------
Main script to train and evaluate models on clean (non-adversarial) data.

Steps:
 1. Load dataset using DataLoader
 2. Create and train model using ModelFactory
 3. Evaluate on test data using evaluator
 4. Save model and metrics to respective folders
"""

import os
from src.utils.helpers import setup_logger
from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.evaluator import evaluate_classifier, save_metrics


def main():
    logger = setup_logger("TrainModel")

    try:
        # Paths
        dataset_cfg = "config/dataset_config.yaml"
        model_cfg = "config/model_config.yaml"
        model_out = "models/trained_model.pkl"
        report_out = "reports/clean_eval_metrics.json"

        logger.info("=== Starting Model Training and Evaluation ===")

        # Step 1: Load and preprocess data
        loader = DataLoader(dataset_cfg)
        df = loader.load_dataset()
        if df is None:
            logger.error("Dataset could not be loaded. Exiting.")
            return

        X_train, X_test, y_train, y_test = loader.preprocess(df)
        if X_train is None:
            logger.error("Preprocessing failed. Exiting.")
            return

        # Step 2: Create and train model
        factory = ModelFactory(model_cfg)
        model = factory.create_model()
        if model is None:
            logger.error("Model creation failed. Exiting.")
            return

        logger.info("Training model...")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Step 3: Evaluate model on clean data
        metrics = evaluate_classifier(model, X_test, y_test)
        logger.info(f"Clean Accuracy: {metrics.get('clean_accuracy')}")

        # Step 4: Save model and metrics
        factory.model = model
        factory.save_model(model_out)
        save_metrics(metrics, report_out)

        logger.info("=== Training & Evaluation completed successfully ===")

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")


if __name__ == "__main__":
    main()
