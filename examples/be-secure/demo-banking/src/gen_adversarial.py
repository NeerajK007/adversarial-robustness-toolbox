"""
gen_adversarial.py
-------------------
Generate adversarial examples via a surrogate-transfer attack and evaluate
transfer success on a saved victim model (XGBoost).

Assumes processed arrays exist under: data/processed/{X_train.npy,X_test.npy,y_train.npy,y_test.npy}
"""

import os
import numpy as np
from src.utils.helpers import setup_logger, load_config
from src.model_factory import ModelFactory
from src.evaluator import evaluate_classifier, save_metrics
from src.attack_manager import AttackManager


LOG = setup_logger("GenAdversarial")

# lazy imports for optional heavy deps
def _ensure_torch_and_art():
    try:
        import torch  # noqa: F401
        from art.estimators.classification import PyTorchClassifier  # noqa: F401
        from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent  # noqa: F401
        return True
    except Exception as e:
        LOG.error("Required libraries for surrogate/ART not available: %s", e)
        return False


def load_processed_data(processed_dir: str):
    """
    Load processed numpy arrays saved by DataLoader.preprocess().
    Returns X_train, X_test, y_train, y_test or (None,...) on failure.
    """
    try:
        X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
        LOG.info("Loaded processed arrays from %s", processed_dir)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        LOG.exception("Failed to load processed arrays: %s", e)
        return None, None, None, None


def load_victim_model(model_cfg_path: str, model_path: str):
    """
    Load victim model using ModelFactory. Returns the model or None.
    """
    mf = ModelFactory(model_cfg_path)
    mf.create_model()  # safe no-op for sklearn models
    model = mf.load_model(model_path, model_type=None)
    if model is None:
        LOG.error("Could not load victim model from %s", model_path)
        return None
    LOG.info("Victim model loaded.")
    return model


def run_hopskipjump_on_victim(victim_model, X_test, y_test, attack_cfg_path):
    """
    Run HopSkipJump directly on the victim (black-box decision attack).
    Returns metrics dict and artifact/report paths.
    """
    am = AttackManager(attack_cfg_path)

    # wrap the victim model (ART XGBoostClassifier)
    try:
        art_clf = am.wrap_model(victim_model)
    except Exception as e:
        LOG.exception("Failed to wrap victim for HopSkipJump: %s", e)
        return {}

    results = {}
    try:
        # generate adv using conservative defaults from config
        x_adv_hsj = am.generate_hopskipjump(art_clf, X_test)
        out_path = am.config.get("output", {}).get("hopskipjump", "artifacts/adv/hopskipjump.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, x_adv=x_adv_hsj)

        metrics = evaluate_classifier(victim_model, x_adv_hsj, y_test)
        report_path = am.config.get("output", {}).get("hopskipjump_report", "reports/hopskipjump_metrics.json")
        save_metrics(metrics, report_path)

        LOG.info("HopSkipJump adversarial accuracy (victim): %s", metrics.get("clean_accuracy"))
        results = {"adv_path": out_path, "report": report_path, "metrics": metrics}
    except Exception as e:
        LOG.exception("HopSkipJump attack/eval failed: %s", e)

    return results



def run_transfer_attack(victim_model, X_train, y_train, X_test, y_test, attack_cfg_path):
    """
    Train a small PyTorch surrogate, craft FGSM/PGD on it, apply to victim,
    evaluate and save results.

    Returns a dict with keys 'fgsm' and 'pgd' containing metrics info.
    """
    if not _ensure_torch_and_art():
        LOG.error("Torch/ART not available; aborting transfer attack.")
        return {}

    # local imports (after availability check)
    import torch
    from src.surrogate import SurrogateTrainer
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    # read attack config
    cfg = load_config(attack_cfg_path) or {}
    attack_cfg = cfg.get("attack", {}) or {}
    surr_cfg = cfg.get("surrogate", {}) or {}

    # surrogate hyperparams (with safe defaults)
    epochs = int(surr_cfg.get("epochs", 3))
    batch_size = int(surr_cfg.get("batch_size", 64))
    lr = float(surr_cfg.get("lr", 1e-3))
    hidden_dim = int(surr_cfg.get("hidden_dim", 64))

    results = {}

    try:
        LOG.info("Training surrogate MLP (epochs=%s, batch=%s, hidden=%s)...", epochs, batch_size, hidden_dim)
        input_dim = X_train.shape[1]
        n_classes = int(len(np.unique(y_train)))
        surr = SurrogateTrainer(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=n_classes, device="cpu")
        surr.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    except Exception as e:
        LOG.exception("Surrogate training failed: %s", e)
        return results

    # Wrap surrogate with ART PyTorchClassifier
    try:
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(surr.model.parameters(), lr=lr)
        art_surrogate = PyTorchClassifier(
            model=surr.model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(input_dim,),
            nb_classes=n_classes,
            device_type="cpu"
        )
        LOG.info("Wrapped surrogate in ART PyTorchClassifier.")
    except Exception as e:
        LOG.exception("Failed to wrap surrogate with ART: %s", e)
        return results

    # FGSM on surrogate -> evaluate on victim
    try:
        fgsm_eps = float(attack_cfg.get("fgsm", {}).get("eps", 0.01))
        LOG.info("Generating FGSM (eps=%s) on surrogate...", fgsm_eps)
        fgsm = FastGradientMethod(estimator=art_surrogate, eps=fgsm_eps)
        x_adv_fgsm = fgsm.generate(x=X_test)
        out_path = cfg.get("output", {}).get("transfer_fgsm", "artifacts/adv/transfer_fgsm.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, x_adv=x_adv_fgsm)
        metrics = evaluate_classifier(victim_model, x_adv_fgsm, y_test)
        report_path = cfg.get("output", {}).get("transfer_fgsm_report", "reports/transfer_fgsm_metrics.json")
        save_metrics(metrics, report_path)
        results["fgsm"] = {"adv_path": out_path, "report": report_path, "metrics": metrics}
        LOG.info("Transfer FGSM adversarial accuracy (victim): %s", metrics.get("clean_accuracy"))
    except Exception as e:
        LOG.exception("Transfer FGSM failed: %s", e)

    # PGD on surrogate -> evaluate on victim
    try:
        pgd_cfg = attack_cfg.get("pgd", {}) or {}
        pgd_eps = float(pgd_cfg.get("eps", 0.03))
        pgd_step = float(pgd_cfg.get("eps_step", 0.01))
        pgd_iter = int(pgd_cfg.get("max_iter", 10))
        LOG.info("Generating PGD (eps=%s, step=%s, iter=%s) on surrogate...", pgd_eps, pgd_step, pgd_iter)
        pgd = ProjectedGradientDescent(estimator=art_surrogate, eps=pgd_eps, eps_step=pgd_step, max_iter=pgd_iter)
        x_adv_pgd = pgd.generate(x=X_test)
        out_path = cfg.get("output", {}).get("transfer_pgd", "artifacts/adv/transfer_pgd.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, x_adv=x_adv_pgd)
        metrics = evaluate_classifier(victim_model, x_adv_pgd, y_test)
        report_path = cfg.get("output", {}).get("transfer_pgd_report", "reports/transfer_pgd_metrics.json")
        save_metrics(metrics, report_path)
        results["pgd"] = {"adv_path": out_path, "report": report_path, "metrics": metrics}
        LOG.info("Transfer PGD adversarial accuracy (victim): %s", metrics.get("clean_accuracy"))
    except Exception as e:
        LOG.exception("Transfer PGD failed: %s", e)

    return results


def main():
    LOG.info("=== Starting transfer adversarial generation & evaluation ===")
    base = os.path.dirname(os.path.dirname(__file__))  # project root
    processed_dir = os.path.join(base, "data", "processed")
    dataset_cfg = os.path.join(base, "config", "dataset_config.yaml")
    model_cfg = os.path.join(base, "config", "model_config.yaml")
    attack_cfg = os.path.join(base, "config", "attack_config.yaml")
    model_path = os.path.join(base, "models", "trained_model.pkl")

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    if X_test is None:
        LOG.error("Processed data not found. Run preprocessing/train first.")
        return

    # ensure dtype compatibility with PyTorch/ART
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    victim = load_victim_model(model_cfg, model_path)
    if victim is None:
        LOG.error("Victim model not available. Aborting.")
        return

    # 1) Transfer attacks (surrogate -> victim)
    results = run_transfer_attack(victim, X_train, y_train, X_test, y_test, attack_cfg) or {}

    # 2) Decision-based HopSkipJump directly on victim
    try:
        # run_hopskipjump_on_victim should be defined in this module (or imported)
        hsj_result = run_hopskipjump_on_victim(victim, X_test, y_test, attack_cfg)
        if hsj_result:
            results["hopskipjump"] = hsj_result
    except NameError:
        LOG.warning("run_hopskipjump_on_victim not found; skipping HopSkipJump step.")
    except Exception as e:
        LOG.exception("HopSkipJump step failed: %s", e)

    # Summary: show adversarial accuracies per attack (if available)
    summary = {k: v.get("metrics", {}).get("clean_accuracy") for k, v in results.items()}
    LOG.info("Transfer + HopSkipJump finished. Summary (adv accuracies): %s", summary)
    LOG.info("=== Transfer adversarial generation & evaluation completed ===")


if __name__ == "__main__":
    main()
