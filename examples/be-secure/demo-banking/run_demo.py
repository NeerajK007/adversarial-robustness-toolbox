"""
run_demo.py
------------
Entry point to execute the full demo pipeline (data load, model train, clean eval).
"""

import os
from src.train_model import main as train_and_eval
from src.utils.helpers import setup_logger

def main():
    logger = setup_logger("RunDemo")
    logger.info("=== Starting Banking Fraud Detection Demo ===")

    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        # Run training and clean evaluation
        train_and_eval()
        logger.info("Demo completed successfully. Check reports/ for metrics.")
    except Exception as e:
        logger.exception(f"Error running demo: {e}")

if __name__ == "__main__":
    main()
