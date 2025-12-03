"""
attack_manager.py
-----------------
Orchestrates adversarial attack generation using IBM ART.

Responsibilities:
- Wrap a scikit-learn / XGBoost model into an ART classifier
- Generate FGSM and PGD adversarial examples (config-driven)
- Optionally clip generated adversarial examples to feature bounds from config
- Save adversarial datasets to disk

Notes:
- This module expects ART to be installed. If ART is not available,
  methods will log an error and raise ImportError.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any
from src.utils.helpers import setup_logger, load_config

logger = setup_logger(__name__)


class AttackManager:
    def __init__(self, config_path: str):
        """
        Initialize AttackManager with a path to attack config.
        The config may include default attack params and optional feature bounds.

        Args:
            config_path: path to YAML config with attack and dataset settings.
        """
        self.config = load_config(config_path)
        self.attack_cfg = self.config.get("attack", {}) or {}
        self.bounds = self.config.get("dataset", {}).get("feature_bounds", None)
        self._art_available = None
        self._ensure_art()

    def _ensure_art(self):
        """
        Lazily check for ART availability and cache result.
        """
        if self._art_available is None:
            try:
                import art  # noqa: F401
                from art.estimators.classification import SklearnClassifier  # noqa: F401
                from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent  # noqa: F401
                self._art_available = True
                logger.info("ART detected and available.")
            except Exception:
                self._art_available = False
                logger.error("IBM ART not available in environment. Install 'adversarial-robustness-toolbox'.")

    def wrap_model(self, model):
        if not self._art_available:
            raise ImportError("ART is not installed")

        try:
            from art.estimators.classification import SklearnClassifier, XGBoostClassifier
            if model.__class__.__name__.lower().startswith("xgb"):
                art_clf = XGBoostClassifier(model=model)
                logger.info("Wrapped model with ART XGBoostClassifier.")
            else:
                art_clf = SklearnClassifier(model=model)
                logger.info("Wrapped model with ART SklearnClassifier.")
            return art_clf
        except Exception as e:
            logger.exception(f"Failed to wrap model: {e}")
            raise



    def _clip_to_bounds(self, x_adv: np.ndarray) -> np.ndarray:
        """
        Clips adversarial examples to configured feature bounds if available.

        The config format for bounds should be something like:
        dataset:
          feature_bounds:
            - [min0, max0]
            - [min1, max1]
            ...
        """
        if not self.bounds:
            return x_adv

        try:
            bounds_arr = np.array(self.bounds, dtype=float)
            mins = bounds_arr[:, 0]
            maxs = bounds_arr[:, 1]
            clipped = np.clip(x_adv, mins, maxs)
            logger.info("Clipped adversarial examples to configured feature bounds.")
            return clipped
        except Exception as e:
            logger.exception(f"Error clipping to bounds: {e}")
            return x_adv

    def generate_fgsm(self, art_clf, x: np.ndarray, eps: float = None) -> np.ndarray:
        """
        Generate FGSM adversarial examples.

        Args:
            art_clf: ART classifier (output of wrap_model)
            x: input feature array (num_samples, num_features)
            eps: L-infinity perturbation budget; if None, read from config

        Returns:
            x_adv: adversarial examples as numpy array
        """
        if not self._art_available:
            raise ImportError("ART is not installed")

        eps = eps if eps is not None else float(self.attack_cfg.get("fgsm", {}).get("eps", 0.01))
        try:
            from art.attacks.evasion import FastGradientMethod
            logger.info("Generating FGSM adversarial examples (eps=%s)...", eps)
            attack = FastGradientMethod(estimator=art_clf, eps=eps)
            x_adv = attack.generate(x=x)
            x_adv = self._clip_to_bounds(x_adv)
            logger.info("FGSM generation complete. Shape: %s", x_adv.shape)
            return x_adv
        except Exception as e:
            logger.exception(f"FGSM generation failed: {e}")
            raise

    def generate_pgd(self, art_clf, x: np.ndarray, eps: float = None, eps_step: float = None,
                     max_iter: int = None) -> np.ndarray:
        """
        Generate PGD adversarial examples.

        Args:
            art_clf: ART classifier (output of wrap_model)
            x: input feature array
            eps: L-inf perturbation bound
            eps_step: step size per iteration
            max_iter: number of iterations

        Returns:
            x_adv: adversarial examples as numpy array
        """
        if not self._art_available:
            raise ImportError("ART is not installed")

        cfg = self.attack_cfg.get("pgd", {}) or {}
        eps = float(eps if eps is not None else cfg.get("eps", 0.03))
        eps_step = float(eps_step if eps_step is not None else cfg.get("eps_step", 0.01))
        max_iter = int(max_iter if max_iter is not None else cfg.get("max_iter", 10))

        try:
            from art.attacks.evasion import ProjectedGradientDescent
            logger.info("Generating PGD adversarial examples (eps=%s, step=%s, iter=%s)...", eps, eps_step, max_iter)
            attack = ProjectedGradientDescent(estimator=art_clf, eps=eps, eps_step=eps_step, max_iter=max_iter)
            x_adv = attack.generate(x=x)
            x_adv = self._clip_to_bounds(x_adv)
            logger.info("PGD generation complete. Shape: %s", x_adv.shape)
            return x_adv
        except Exception as e:
            logger.exception(f"PGD generation failed: {e}")
            raise

    def generate_hopskipjump(self, art_clf, x: np.ndarray, max_iter: int = None, max_eval: int = None) -> np.ndarray:
        """
        Generate adversarial examples using HopSkipJump (decision-based).
        Works directly with non-differentiable classifiers (trees).

        Args:
            art_clf: ART classifier wrapped for the victim model (e.g., XGBoostClassifier)
            x: input array (n_samples, n_features)
            max_iter: maximum iterations (controls search depth)
            max_eval: maximum number of model queries per iteration (ART param)

        Returns:
            x_adv: adversarial examples array
        """
        if not self._art_available:
            raise ImportError("ART is not installed")

        cfg = self.attack_cfg.get("hopskipjump", {}) or {}
        max_iter = int(max_iter if max_iter is not None else cfg.get("max_iter", 10))
        max_eval = int(max_eval if max_eval is not None else cfg.get("max_eval", 100))

        try:
            from art.attacks.evasion import HopSkipJump
            logger.info("Generating HopSkipJump (iter=%s, max_eval=%s)...", max_iter, max_eval)
            attack = HopSkipJump(classifier=art_clf, max_iter=max_iter, max_eval=max_eval)
            x_adv = attack.generate(x=x)
            x_adv = self._clip_to_bounds(x_adv)
            logger.info("HopSkipJump generation complete. Shape: %s", x_adv.shape)
            return x_adv
        except Exception as e:
            logger.exception(f"HopSkipJump generation failed: {e}")
            raise


    def save_adversarial(self, x_adv: np.ndarray, out_path: str):
        """
        Save adversarial examples as a numpy .npz file.

        Args:
            x_adv: adversarial array
            out_path: destination path (.npz)
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, x_adv=x_adv)
            logger.info("Saved adversarial examples to %s", out_path)
        except Exception as e:
            logger.exception(f"Failed to save adversarial examples: {e}")
