"""
helpers.py
-----------
Shared utilities for demo-banking project.
Includes:
 - Centralized logger setup
 - Config loader for YAML files
"""

import os
import yaml
import logging

def setup_logger(name: str, log_file: str = "demo_banking.log", level=logging.INFO):
    """
    Creates and returns a logger instance with standard formatting.
    Logs both to file and console.

    Args:
        name (str): Name of the logger.
        log_file (str): File path to log output.
        level: Logging level (default: INFO)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Prevent duplicate handlers
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_config(config_path: str) -> dict:
    """
    Loads and returns configuration data from a YAML file.

    Args:
        config_path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration data.
    """
    logger = setup_logger(__name__)
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
    except Exception as e:
        logger.exception(f"Error loading config from {config_path}: {e}")
        return {}
