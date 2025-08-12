# Imports
import os
import numpy as np
import tensorflow as tf

# Configure TensorFlow for ART compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from art.estimators.classification import TensorFlowV2Classifier
import argparse
import json
from datetime import datetime

def load_classifier_model(model_path):
    """Load and validate the classifier model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Ensure model is compiled
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model compiled with default settings")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def prepare_data(num_samples=100, binary_classes=None):
    """Prepare MNIST data, optionally filtering for binary classification."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    if binary_classes is not None:
        # Filter for binary classification
        mask = np.isin(y_test, binary_classes)
        x_test = x_test[mask]
        y_test = y_test[mask]
        
        # Remap labels to 0 and 1
        y_test = np.where(y_test == binary_classes[0], 0, 1)
        num_classes = 2
    else:
        num_classes = 10
    
    # Take sample
    x_sample = x_test[:num_samples]
    y_sample_labels = y_test[:num_samples]
    y_sample_cat = to_categorical(y_sample_labels, num_classes)
    
    print(f"Data prepared: {len(x_sample)} samples, {num_classes} classes")
    print(f"Sample shape: {x_sample.shape}")
    return x_sample, y_sample_cat, y_sample_labels

from tensorflow.keras.losses import CategoricalCrossentropy

def create_art_classifier(model):
    """Create ART classifier using TensorFlowV2Classifier."""
    try:
        # Re-enable eager execution for compatibility
        tf.compat.v1.enable_eager_execution()

        loss_object = CategoricalCrossentropy()

        classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=model.output_shape[-1],
            input_shape=model.input_shape[1:],
            loss_object=loss_object,
            clip_values=(0.0, 1.0)
        )
        print("ART TensorFlowV2Classifier created successfully")
        return classifier
    except Exception as e:
        print(f"Failed to create TensorFlowV2Classifier: {e}")
        # Fallback to KerasClassifier
        try:
            from art.estimators.classification import KerasClassifier
            classifier = KerasClassifier(
                model=model,
                clip_values=(0.0, 1.0),
                use_logits=False
            )
            print("Fallback to KerasClassifier successful")
            return classifier
        except Exception as e2:
            print(f"KerasClassifier also failed: {e2}")
            return ModelWrapper(model)

class ModelWrapper:
    """Simple wrapper for model prediction if ART classifier fails."""
    def __init__(self, model):
        self.model = model
        
    def predict(self, x):
        return self.model.predict(x, verbose=0)

def run_attack(attack_class, attack_params, classifier, x, y, name, y_cat=None):
    """Run a single adversarial attack and return results."""
    print(f"\nRunning {name}...")
    try:
        # Create attack instance
        attack = attack_class(estimator=classifier, **attack_params)
        
        # Generate adversarial examples
        if y_cat is not None:
            x_adv = attack.generate(x=x, y=y_cat)
        else:
            x_adv = attack.generate(x=x)
        
        if x_adv is None:
            raise ValueError("Attack failed to generate adversarial examples")
        
        # Calculate accuracy on adversarial examples
        preds = classifier.predict(x_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) / len(y)
        
        # Calculate perturbation statistics
        perturbation = np.abs(x_adv - x)
        avg_perturbation = np.mean(perturbation)
        max_perturbation = np.max(perturbation)
        l2_perturbation = np.mean(np.sqrt(np.sum(perturbation**2, axis=(1,2,3))))
        
        print(f"Accuracy on {name} adversarial examples: {acc:.4f}")
        print(f"Average L∞ perturbation: {avg_perturbation:.6f}")
        print(f"Average L2 perturbation: {l2_perturbation:.6f}")
        print(f"Max perturbation: {max_perturbation:.6f}")
        
        return {
            'accuracy': acc,
            'avg_perturbation': avg_perturbation,
            'max_perturbation': max_perturbation,
            'l2_perturbation': l2_perturbation,
            'success': True
        }, x_adv
        
    except Exception as e:
        print(f"{name} failed: {e}")
        return {
            'accuracy': None,
            'avg_perturbation': None,
            'max_perturbation': None,
            'l2_perturbation': None,
            'success': False,
            'error': str(e)
        }, None

def get_attack_configurations(attack_type='fast'):
    """Get attack configurations based on type."""
    try:
        from art.attacks.evasion import (
            FastGradientMethod, 
            BasicIterativeMethod, 
            ProjectedGradientDescent,
            DeepFool,
            CarliniL2Method
        )
        
        if attack_type == 'fast':
            return [
                (FastGradientMethod, {"eps": 0.1, "norm": np.inf}, "FGSM_L∞"),
                (FastGradientMethod, {"eps": 0.2, "norm": np.inf}, "FGSM_L∞_0.2"),
                (BasicIterativeMethod, {"eps": 0.1, "max_iter": 7}, "BIM"),
                (ProjectedGradientDescent, {"eps": 0.1, "max_iter": 7}, "PGD"),
            ]
        elif attack_type == 'comprehensive':
            attacks = [
                (FastGradientMethod, {"eps": 0.1, "norm": np.inf}, "FGSM_L∞_0.1"),
                (FastGradientMethod, {"eps": 0.2, "norm": np.inf}, "FGSM_L∞_0.2"),
                (FastGradientMethod, {"eps": 0.3, "norm": np.inf}, "FGSM_L∞_0.3"),
                (BasicIterativeMethod, {"eps": 0.1, "max_iter": 10}, "BIM"),
                (ProjectedGradientDescent, {"eps": 0.1, "max_iter": 10}, "PGD_L∞"),
                (ProjectedGradientDescent, {"eps": 0.2, "max_iter": 10}, "PGD_L∞_0.2"),
                (DeepFool, {"max_iter": 10}, "DeepFool"),
            ]
            
            # Try to add C&W attack if available
            try:
                attacks.append((CarliniL2Method, {"max_iter": 10, "confidence": 0.0}, "C&W_L2"))
            except Exception:
                pass
                
            return attacks
        else:  # custom
            return [
                (FastGradientMethod, {"eps": 0.05, "norm": np.inf}, "FGSM_eps0.05"),
                (FastGradientMethod, {"eps": 0.1, "norm": np.inf}, "FGSM_eps0.1"),
                (FastGradientMethod, {"eps": 0.15, "norm": np.inf}, "FGSM_eps0.15"),
                (FastGradientMethod, {"eps": 0.2, "norm": np.inf}, "FGSM_eps0.2"),
                (FastGradientMethod, {"eps": 0.3, "norm": np.inf}, "FGSM_eps0.3"),
                (BasicIterativeMethod, {"eps": 0.1, "max_iter": 5}, "BIM_5iter"),
                (BasicIterativeMethod, {"eps": 0.1, "max_iter": 10}, "BIM_10iter"),
                (ProjectedGradientDescent, {"eps": 0.1, "max_iter": 10}, "PGD_10iter"),
                (ProjectedGradientDescent, {"eps": 0.1, "max_iter": 20}, "PGD_20iter"),
                (DeepFool, {"max_iter": 10}, "DeepFool_10iter"),
                (DeepFool, {"max_iter": 20}, "DeepFool_20iter"),
            ]
    except ImportError as e:
        print(f"Some attacks not available: {e}")
        # Fallback to basic attacks
        from art.attacks.evasion import FastGradientMethod
        return [
            (FastGradientMethod, {"eps": 0.1}, "FGSM_0.1"),
            (FastGradientMethod, {"eps": 0.2}, "FGSM_0.2"),
        ]

def save_results(results, output_file, metadata=None):
    """Save results to JSON file."""
    timestamp = datetime.now().isoformat()
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for name, result in results.items():
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.floating):
                serializable_result[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_result[key] = int(value)
            else:
                serializable_result[key] = value
        serializable_results[name] = serializable_result
    
    output_data = {
        'timestamp': timestamp,
        'metadata': metadata or {},
        'results': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_file}")

def visualize_results(results, save_path=None):
    """Create visualization of attack results."""
    successful_attacks = [(name, res['accuracy']) for name, res in results.items() 
                         if res['success'] and res['accuracy'] is not None]
    
    if not successful_attacks:
        print("No successful attacks to visualize.")
        return
    
    names, accuracies = zip(*successful_attacks)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(names)), accuracies)
    plt.xlabel('Attack Method', fontsize=12)
    plt.ylabel('Accuracy on Adversarial Examples', fontsize=12)
    plt.title('Model Robustness Against Adversarial Attacks\n(Lower accuracy indicates more successful attack)', fontsize=14)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add reference lines
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10% accuracy (very vulnerable)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% accuracy (vulnerable)')
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% accuracy (robust)')
    
    # Color bars based on accuracy
    for bar, acc in zip(bars, accuracies):
        if acc < 0.1:
            bar.set_color('darkred')
        elif acc < 0.3:
            bar.set_color('red')
        elif acc < 0.5:
            bar.set_color('orange')
        elif acc < 0.7:
            bar.set_color('yellow')
        else:
            bar.set_color('green')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Red Team Classifier with Adversarial Attacks')
    parser.add_argument('--model-path', required=True, help='Path to the trained model file')
    parser.add_argument('--attack-type', choices=['fast', 'comprehensive', 'custom'], 
                       default='fast', help='Type of attack suite to run')
    parser.add_argument('--num-samples', type=int, default=100, 
                       help='Number of test samples to use')
    parser.add_argument('--binary-classes', nargs=2, type=int, default=None,
                       help='Two classes for binary classification (e.g., --binary-classes 0 1)')
    parser.add_argument('--output-dir', default='./results', 
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualization of results')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.debug:
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Eager execution enabled: {tf.executing_eagerly()}")
        
        # Load model
        print("Loading model...")
        classifier_model = load_classifier_model(args.model_path)
        
        # Create ART classifier
        print("Creating ART classifier...")
        classifier = create_art_classifier(classifier_model)
        
        # Prepare data
        print("Preparing data...")
        x_sample, y_sample, y_sample_labels = prepare_data(
            num_samples=args.num_samples, 
            binary_classes=args.binary_classes
        )
        
        # Test classifier on clean data first
        print("\nTesting on clean data...")
        clean_preds = classifier.predict(x_sample)
        clean_acc = np.sum(np.argmax(clean_preds, axis=1) == np.argmax(y_sample, axis=1)) / len(y_sample)
        print(f"Clean accuracy: {clean_acc:.4f}")
        
        # Get attack configurations
        attack_list = get_attack_configurations(args.attack_type)
        
        # Run attacks
        results = {}
        print(f"\nRunning {len(attack_list)} attacks...")
        
        for i, (attack_class, params, name) in enumerate(attack_list, 1):
            print(f"\n[{i}/{len(attack_list)}] Processing {name}...")
            result, x_adv = run_attack(attack_class, params, classifier, x_sample, y_sample, name, y_cat=y_sample)
            results[name] = result
        
        # Print summary
        print("\n" + "="*80)
        print("ATTACK RESULTS SUMMARY")
        print("="*80)
        print(f"Clean Accuracy: {clean_acc:.4f}")
        print("-" * 80)
        print(f"{'Attack Name':<30} {'Accuracy':<10} {'Avg Pert':<12} {'Max Pert':<12} {'Status'}")
        print("-" * 80)
        
        successful_attacks = 0
        for name, result in results.items():
            if result['success']:
                print(f"{name:<30} {result['accuracy']:<10.4f} {result['avg_perturbation']:<12.6f} {result['max_perturbation']:<12.6f} SUCCESS")
                if result['accuracy'] < 0.5:
                    successful_attacks += 1
            else:
                print(f"{name:<30} {'N/A':<10} {'N/A':<12} {'N/A':<12} FAILED")
        
        print("-" * 80)
        print(f"Successful attacks (accuracy < 50%): {successful_attacks}/{len([r for r in results.values() if r['success']])}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'model_path': args.model_path,
            'attack_type': args.attack_type,
            'num_samples': args.num_samples,
            'binary_classes': args.binary_classes,
            'clean_accuracy': clean_acc,
            'tensorflow_version': tf.__version__
        }
        
        results_file = os.path.join(args.output_dir, f'attack_results_{timestamp}.json')
        save_results(results, results_file, metadata)
        
        # Generate visualization if requested
        if args.visualize:
            viz_file = os.path.join(args.output_dir, f'attack_results_{timestamp}.png')
            visualize_results(results, viz_file)
        
        print(f"\nRed teaming assessment completed!")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())