#!/usr/bin/env python3
"""
evaluate_robustness.py
Batch evaluator for model robustness against adversarial attacks.
Uses existing loaders and attack functions.
"""

import json
import torch
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import time
from torchvision import transforms
from PIL import Image


import os

from torch.utils.data import DataLoader, Subset
from datasets.mvtec_loader import MVTecSyntheticDataset
from datasets.traffic_sign_loader import TrafficSignDataset

from model_loader import load_demo_model
from model_registry import get_model_cfg


import logging

logging.basicConfig(
    level=logging.INFO,
    format="==== [%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] ➜ %(message)s ====",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("ART-Eval")

# -------------------------------
# Config (hardcoded for now)
# -------------------------------
def get_config():
    config = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "demo_type":"Indian-trafic-signal-misclassification",
        "attacks": [
            #{ "name": "fgsm", "parameters": { "eps": 0.1 } },
            #{ "name": "pgd", "parameters": { "eps": 0.01, "eps_step": 0.01, "max_iter": 10 } },
            #{ "name": "pgd", "parameters": { "eps": 0.03, "eps_step": 0.01, "max_iter": 10 } },
            { "name": "pgd", "parameters": { "eps": 0.1, "eps_step": 0.01, "max_iter": 10 } },
            #{ "name": "cw", "parameters": { "confidence": 0.5, "max_iter": 1 } }
        ],
        "run_options": {
            "batch_size": 8,
            "limit": 5,
            "detailed_logs": False
        },
        "output": {
            "report_dir": "reports/"
        }
    }
    return config


# -------------------------------
# Core evaluation pipeline
# -------------------------------
def load_model(model_cfg: dict, device):
    """
    Unified model loader (delegates to model_loader.py)
    """
    demo_type = model_cfg.get("demo_type")
    variant = model_cfg.get("variant", "normal")
    return load_demo_model(demo_type, device, variant)


def load_dataset(demo_type, batch_size, limit=None):
    """
    Load dataset for the given demo_type and wrap in DataLoader.

    Returns:
        dataloader: torch.utils.data.DataLoader
        dataset_meta: dict with basic dataset info
    """
  
    # Project root relative to this script
    project_root = os.path.dirname(__file__)
    data_root = os.path.join(project_root, "data")

   
    logger.info(f"demo_type: {demo_type.lower()}")
    # Select dataset based on demo_type
    if "trafic" in demo_type.lower():
        logger.info(f"Calling TrafficSignDataset from datasets/")
        dataset_path = os.path.join(data_root, "1_Indian_traffic_sign_classification_dataset_class4", "test")
        dataset = TrafficSignDataset(dataset_path, is_train=False)
    elif "tile" in demo_type.lower() or "mvtec" in demo_type.lower():
        logger.info(f"Calling MVTecSyntheticDataset from datasets/")
        dataset_path = os.path.join(data_root, "mvtec_anomaly_detection")
        dataset = MVTecSyntheticDataset(dataset_path, is_train=False)
    else:
        raise ValueError(f"Unsupported demo_type: {demo_type}")

    # Apply optional limit for faster debugging
    if limit and limit < len(dataset):
        indices = list(range(limit))
        dataset = Subset(dataset, indices)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Build dataset metadata
    dataset_meta = {
        "name": demo_type,
        "size_test": len(dataset),
        "num_classes": getattr(dataset, "num_classes", None) or "unknown"
    }

    logger.info(f"Loaded dataset '{demo_type}' with {len(dataset)} samples.")
    
    return dataloader, dataset_meta


def evaluate_clean(model, dataloader, device):
    """
    Evaluate model on clean (unperturbed) test data.
    Returns a dictionary with overall accuracy, avg confidence, and per-class results.
    """
  

    model.eval()

    total, correct = 0, 0
    confidences = []
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            confidences.extend(conf.cpu().numpy())

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[t] = per_class_total.get(t, 0) + 1
                if t == p:
                    per_class_correct[t] = per_class_correct.get(t, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    per_class_results = {
        str(cls): round(per_class_correct.get(cls, 0) / per_class_total.get(cls, 1), 4)
        for cls in per_class_total.keys()
    }

    result = {
        "clean_accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_conf, 4),
        "per_class_accuracy": per_class_results,
    }

    logger.info(f"Clean Evaluation: accuracy={accuracy:.4f}, avg_conf={avg_conf:.4f}")
    return result


def run_attack(model, dataloader, device, attack_name, attack_params):
    """
    Generate adversarial examples (per-sample) using demo_attack.generate_adversarial
    and evaluate model performance on those adversarial images.

    Returns:
      {
        "attack": attack_name,
        "parameters": attack_params,
        "metrics": {
           "accuracy_after_attack": float,
           "avg_confidence_after_attack": float,
           "attack_success_rate": float,
           "per_class_accuracy": { class_idx: acc, ... }
        },
        "samples_evaluated": int,
        "duration_sec": float
      }
    Notes:
      - This implementation converts batch tensors back to PIL images, calls the
        existing generate_adversarial(...) (which returns a PIL.Image), re-applies
        the model's input transforms, and runs inference on the adversarial image.
      - It's simple and robust (works with your current demo_attack functions),
        but will be slower than a pure-batch ART generation approach.
    """



    # reuse the same normalization used by your datasets / model input
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    post_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    to_pil = transforms.ToPILImage()

    # lazy import of your attack wrapper to avoid circular imports at module load
    from demo_attack import generate_adversarial

    model.eval()
    start_ts = time.time()

    total = 0
    correct_after = 0
    confidences_after = []

    per_class_total = {}
    per_class_correct = {}

    # iterate dataset (no shuffle)
    for images, labels in dataloader:
        # images: tensor already normalized by dataset transforms
        # we need to convert each tensor back to a PIL image (unnormalize)
        batch_size = images.size(0)

        # Move tensors to CPU for conversion & to device for inference
        images_cpu = images.cpu()
        labels_cpu = labels.cpu().numpy()

        for i in range(batch_size):
            img_t = images_cpu[i]  # normalized tensor [C,H,W]
            # un-normalize to [0,1] range for PIL conversion
            unnorm = img_t.clone()
            for c in range(3):
                unnorm[c] = unnorm[c] * std[c] + mean[c]
            unnorm = torch.clamp(unnorm, 0.0, 1.0)

            pil_img = to_pil(unnorm)  # PIL.Image

            try:
                # generate adversarial image (PIL) using your demo_attack wrapper
                adv_result = generate_adversarial(pil_img, model, device,
                                                 attack_name=attack_name,
                                                 attack_params=attack_params)
                adv_pil = adv_result.get("adversarial_image", None)
                if adv_pil is None:
                    logger.warning(f"No adversarial image returned for sample idx={total+i}")
                    continue
            except Exception as e:
                logger.error(f"Attack generation failed for sample idx={total+i}: {e}", exc_info=True)
                continue

            # convert adv_pil back to tensor using same post_transform used for model input
            adv_tensor = post_transform(adv_pil).unsqueeze(0).to(device)  # [1,C,H,W]
            lbl = int(labels_cpu[i])

            # predict on adv image
            with torch.no_grad():
                outputs = model(adv_tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)
                pred_item = int(pred.item())
                conf_item = float(conf.item())

            # aggregate
            total += 1
            if pred_item == lbl:
                correct_after += 1
                per_class_correct[lbl] = per_class_correct.get(lbl, 0) + 1
            per_class_total[lbl] = per_class_total.get(lbl, 0) + 1
            confidences_after.append(conf_item)

    duration = time.time() - start_ts
    accuracy_after = (correct_after / total) if total > 0 else 0.0
    avg_conf_after = float(np.mean(confidences_after)) if confidences_after else 0.0
    # attack success rate = fraction of samples that were misclassified after attack
    attack_success_rate = 1.0 - accuracy_after

    # per-class accuracy
    per_class_accuracy = {}
    for cls, tot in per_class_total.items():
        per_class_accuracy[str(cls)] = round(per_class_correct.get(cls, 0) / tot, 4)

    adv_results = {
        "attack": attack_name,
        "parameters": attack_params,
        "metrics": {
            "accuracy_after_attack": round(accuracy_after, 4),
            "avg_confidence_after_attack": round(avg_conf_after, 4),
            "attack_success_rate": round(attack_success_rate, 4),
            "per_class_accuracy": per_class_accuracy
        },
        "samples_evaluated": total,
        "duration_sec": round(duration, 3)
    }

    logger.info(f"Completed attack '{attack_name}': samples={total}, acc={adv_results['metrics']['accuracy_after_attack']}, asr={adv_results['metrics']['attack_success_rate']}, duration={adv_results['duration_sec']}s")
    return adv_results


def aggregate_metrics(clean_results, adv_results, dataset_meta=None, model_cfg=None, attack_cfg=None, run_id=None):
    """
    Compute global and per-class metrics (accuracy, ASR, confidence drop).
    Combines clean and adversarial results into a unified summary block.
    """
    # implement or placeholder
    summary = {
        "attack": attack_cfg.get("name") if attack_cfg else "unknown",
        "metrics": {
            "clean_accuracy": clean_results.get("clean_accuracy", 0),
            "adv_accuracy": adv_results["metrics"].get("accuracy_after_attack", 0),
            "attack_success_rate": adv_results["metrics"].get("attack_success_rate", 0),
            "avg_confidence_drop": round(
                clean_results.get("avg_confidence", 0) -
                adv_results["metrics"].get("avg_confidence_after_attack", 0), 4
            )
        },
        "per_class_comparison": adv_results["metrics"].get("per_class_accuracy", {}),
        "samples_evaluated": adv_results.get("samples_evaluated", 0)
    }
    return summary


def save_report(report, config):
    """
    Transform internal ART evaluation results into the standardized report format.
    Prints the JSON-formatted report to console instead of saving to file.
    """
    import json
    from datetime import datetime

    # Pull metadata
    meta = report.get("REPORT_META", {})
    attacks = report.get("EVALUATION_CONFIG", {}).get("attacks", [])
    summaries = report.get("ATTACK_SUMMARIES", [])
    clean_stats = report.get("CLEAN_RESULTS", {})

    # --- Build new structure ---
    final_report = {
        "REPORT_META": {
            "report_id": meta.get("report_id"),
            "report_type": "adversarial_robustness",
            "generated_at": meta.get("generated_at", datetime.now().isoformat()),
            "dataset": {
                "name": meta.get("model", {}).get("dataset_name"),
                #"name": meta.get("dataset_meta", {}).get("name"),
                "size_test": meta.get("dataset_meta", {}).get("size_test"),
                "num_classes": meta.get("dataset_meta", {}).get("num_classes"),
                "source":"datasets url"
            },
            "model": {
                "name": meta.get("model", {}).get("model_fn"),
                "pretrained_on":"ImageNet",
                "fine_tuned_on":meta.get("model", {}).get("dataset_name"),
                "variant": meta.get("model", {}).get("variant"),
                "weights_file": meta.get("model", {}).get("weights_path"),
                "source":"https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
            },
            "environment": {
                "framework": "PyTorch 2.1",
                "art_version": "1.20.1",
                "device": meta.get("environment", {}).get("torch_device", "cpu"),
            },
        },
        "EVALUATION_CONFIG": {
            "Evasion": {
                "attacks": attacks
            },
            "Data Poisoning": {
                "attacks": [
                    {
                        "name": "backdoor_attack",
                        "parameters": "N/A"
                    }
                ]
            },
            "metrics_collected": [
                "clean_accuracy",
                "adv_accuracy",
                "attack_success_rate",
                "avg_confidence_drop"
            ]
        },
        "GLOBAL_RESULTS": {
            "clean_accuracy": clean_stats.get("clean_accuracy", 0),
            "avg_confidence": clean_stats.get("avg_confidence", 0),
            "Evasion": {
                "attacks": [
                    {
                        "name": s["attack"],
                        "parameters": s["metadata"].get("attack_parameters", {}),
                        "adv_accuracy": s["metrics"].get("adv_accuracy"),
                        "attack_success_rate":s["metrics"].get("attack_success_rate"),
                        "avg_confidence_drop": s["metrics"].get("avg_confidence_drop")
                       
                    }
                    for s in summaries
                ]
            },
            "Data Poisoning": {}
        }
    }

    # --- Print the report prettily ---
    print("\n\n========== FINAL FORMATTED REPORT ==========")
   # remove unserializable entries
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if not callable(v)}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif callable(obj):
            return str(obj.__name__)  # e.g., "mobilenet_v2"
        else:
            return obj

    clean_report = clean_for_json(final_report)
    print(json.dumps(clean_report, indent=2))
    print("============================================\n")


# -------------------------------
# Main
# -------------------------------
def main():
    """
    Orchestrator for a single model run:
      - load config
      - load model & dataset
      - evaluate clean
      - run each attack
      - aggregate metrics
      - save report
    """
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = config.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_base = config.get("output", {}).get("report_dir", "reports")
    logger.info(f"Starting robustness evaluation run_id={run_id} on device={device}")
    logger.info(f"Reports will be saved under: {output_base}")

    # Pick the first (or only) model from config
   # model_cfg = config["models"][1]
    model_cfg = get_model_cfg(config.get("demo_type"))
    model_name = model_cfg.get("model_fn", "unnamed_model")
    demo_type = model_cfg.get("demo_type")
    variant = model_cfg.get("variant", "normal")
    logger.info(f"Processing model: {model_name} (demo_type={demo_type}, variant={variant})")

    try:
        # 1) Load model
        model = load_model(model_cfg, device)
        logger.info(f"Model load complete.")

        # 2) Load dataset
        run_opts = config.get("run_options", {})
        batch_size = run_opts.get("batch_size", 16)
        limit = run_opts.get("limit", None)
        dataloader, dataset_meta = load_dataset(demo_type, batch_size=batch_size, limit=limit)

        # 3) Evaluate clean baseline
        logger.info(f"Evaluating clean performance for {model_name} on {demo_type} ...")
        clean_results = evaluate_clean(model, dataloader, device)
        
        logger.info(f"Clean Result: {clean_results}")

        # 4) For each attack, run and aggregate
        attacks = config.get("attacks", [])
        all_attack_summaries = []
        for attack_cfg in attacks:
            attack_name = attack_cfg.get("name")
            attack_params = attack_cfg.get("parameters", {})

            logger.info(f"Running attack '{attack_name}' with params={attack_params} ...")
            adv_results = run_attack(model, dataloader, device, attack_name, attack_params)

            summary = aggregate_metrics(clean_results, adv_results,
                                        dataset_meta=dataset_meta,
                                        model_cfg=model_cfg,
                                        attack_cfg=attack_cfg,
                                        run_id=run_id)
            summary["metadata"] = {
                "model_name": model_name,
                "demo_type": demo_type,
                "variant": variant,
                "attack": attack_name,
                "attack_parameters": attack_params
            }
            all_attack_summaries.append(summary)

        # 5) Build final report
        report = {
            "REPORT_META": {
                "report_id": run_id,
                "generated_at": datetime.now().isoformat(),
                "model": model_cfg,
                "dataset_meta": dataset_meta,
                "environment": {"torch_device": str(device)}
            },
            "EVALUATION_CONFIG": {
                "attacks": config.get("attacks", []),
                "run_options": run_opts
            },
            "ATTACK_SUMMARIES": all_attack_summaries
        }
        
       # logger.info(f"report: {report}")

        # 6) Save report
        report["CLEAN_RESULTS"] = clean_results
        save_report(report, config)

    except Exception as e:
        logger.error("Exception while processing model %s: %s", model_name, e, exc_info=True)

    logger.info(f"Robustness evaluation run completed.")



if __name__ == "__main__":
    main()
