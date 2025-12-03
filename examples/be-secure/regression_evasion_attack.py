import os
import joblib
import argparse
import logging
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from datetime import datetime


from art.estimators.regression import ScikitlearnRegressor
from art.estimators.regression import KerasRegressor
from art.estimators.classification import TensorFlowV2Classifier

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import HopSkipJump



import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def fgsm_attack(model, X_test, modelType, epsilon):
    if modelType == "sklearn":
        regressor = ScikitlearnRegressor(model=model)
    elif modelType == "tf":
        regressor = KerasRegressor(model=model)
    else:
        raise ValueError("Unsupported model type")
    attack = FastGradientMethod(estimator=regressor, eps=epsilon)
    X_adv = attack.generate(x=X_test)
    return X_adv


def bim_attack(model, X_test, modelType, epsilon):
    if modelType == "sklearn":
        regressor = ScikitlearnRegressor(model=model)
    elif modelType == "tf":
        regressor = KerasRegressor(model=model)
    else:
        raise ValueError("Unsupported model type")
    attack = BasicIterativeMethod(estimator=regressor, eps=epsilon)
    X_adv = attack.generate(x=X_test)
    return X_adv

def pgd_attack(model, X_test, modelType, epsilon, eps_step=0.01, max_iter=10):
    epsilon = float(epsilon)
    eps_step = float(eps_step)
    if modelType == "sklearn":
        regressor = ScikitlearnRegressor(model=model)
    elif modelType == "tf":
        regressor = KerasRegressor(model=model)
    else:
        raise ValueError("Unsupported model type")
    attack = ProjectedGradientDescent(estimator=regressor, eps=epsilon, eps_step=eps_step, max_iter=max_iter)
    X_adv = attack.generate(x=X_test)
    return X_adv

def hopskipjump_attack(model, X_test, modelType, max_iter):
    if modelType == "sklearn":
        regressor = ScikitlearnRegressor(model=model)
    elif modelType == "tf":
        regressor = KerasRegressor(model=model)
    else:
        raise ValueError("Unsupported model type")
    attack = HopSkipJump(classifier=regressor, max_iter=max_iter)
    X_adv = attack.generate(x=X_test)
    return X_adv


def run_evasion_attack(model, X_test, y_test, modelType, epsilons):
    results = {}
    
    try:
        for eps in epsilons:
            # PGD attack
            X_adv_pgd = pgd_attack(model, X_test, modelType, eps)
            mse_adv_pgd = np.mean((model.predict(X_adv_pgd) - y_test) ** 2)
            logging.info(f"[PGD] MSE on adversarial data (epsilon={eps}): {mse_adv_pgd:.4f}")
            results[f"PGD_{eps}"] = mse_adv_pgd

            # FGSM attack
            X_adv_fgsm = fgsm_attack(model, X_test, modelType, eps)
            mse_adv_fgsm = np.mean((model.predict(X_adv_fgsm) - y_test) ** 2)
            logging.info(f"[FGSM] MSE on adversarial data (epsilon={eps}): {mse_adv_fgsm:.4f}")
            results[f"FGSM_{eps}"] = mse_adv_fgsm
            
            # BIM attack
            X_adv_bim = bim_attack(model, X_test, modelType, epsilon=eps)
            mse_adv_bim = np.mean((model.predict(X_adv_bim) - y_test) ** 2)
            logging.info(f"[BIM] MSE on adversarial data (epsilon={eps}): {mse_adv_bim:.4f}")
            results[f"BIM_{eps}"] = mse_adv_bim
        
        X_adv_hsj = hopskipjump_attack(model, X_test, modelType, max_iter=10)
        mse_adv_hsj = np.mean((model.predict(X_adv_hsj) - y_test) ** 2)
        logging.info(f"[HSJ] MSE on adversarial data: {mse_adv_hsj:.4f}")
        results["HSJ"] = mse_adv_hsj
        
    except Exception as e:
        logging.info(f"=============Exception==================={e}")
    
    return results


def create_and_save_sk_model(model_file, X_train=None, y_train=None):
    if os.path.exists(model_file):
        logging.info(f"Model already exists at {model_file}")
        return

    if X_train is None or y_train is None:
        X_train, y_train = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        logging.info("Generated synthetic regression data for training.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    logging.info(f"Model trained and saved at {model_file}")


def create_and_save_tf_model(model_path, model_file, X_train=None, y_train=None):
    """
    Train a TensorFlow regression model and save it to disk.
    """
    if model_file is not None:
        logging.info(f"======================== TensorFlow model already exists at {model_file}")
        return model_file

    if X_train is None or y_train is None:
        X_train, y_train = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        logging.info("======================== Generated synthetic regression data for TensorFlow training.")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    model_path = os.path.join(model_path, 'tf_reg_model.h5')
    model.save(model_path)
    logging.info(f"======================== TensorFlow model trained and saved at {model_path}")
    
    return model_path


def load_model(modelFile, modelType):
    print("======================== load_model - modelFile: ", modelFile)
    if modelType == "sklearn":
        return joblib.load(modelFile)
    elif modelType == "tf":
        return tf.keras.models.load_model(modelFile)
    else:
        raise ValueError("Unsupported model type")


def main():
    parser = argparse.ArgumentParser(
        description="Regression Adversarial Scan Playbook: Simulate attacks on regression models."
    )
    
    modelType = "tf"

    model_file = os.environ.get("MODEL_FILE")
    model_path = os.environ.get("MODEL_PATH")
    if not model_file:
        model_file = None
    if not model_path:
        raise RuntimeError("Please set MODEL_PATH using export MODEL_PATH=<path_to_model>")

    modelFile = ""
    if modelType == "tf":
        modelFile = create_and_save_tf_model(model_path, model_file)
        print("======================== modelFile PATH : ", modelFile)
    else:
        create_and_save_sk_model(model_path, model_file)
        
    model = load_model(modelFile, modelType)

    # Generate synthetic test data (or load your real test data here)
    X_test, y_test = make_regression(n_samples=50, n_features=10, noise=0.1, random_state=123)
    logging.info("======================== Generated synthetic regression data for testing.")

    # Define a list of epsilon values
    epsilons = [0.01, 0.05, 0.1, 0.2]

    # Run evasion attack for each epsilon
    results = run_evasion_attack(model, X_test, y_test, modelType, epsilons)
    
    logging.info("======================== Attack results:")

    print("PGD Attack Results:")
    for eps in epsilons:
        key = f"PGD_{eps}"
        mse = results.get(key)
        if mse is not None:
            print(f"PGD (epsilon={eps}) ===================== MSE on adversarial data: {mse:.4f}")
        else:
            print(f"PGD (epsilon={eps}) ===================== MSE on adversarial data: N/A")

    print("\nFGSM Attack Results:")
    for eps in epsilons:
        key = f"FGSM_{eps}"
        mse = results.get(key)
        if mse is not None:
            print(f"FGSM (epsilon={eps}) ===================== MSE on adversarial data: {mse:.4f}")
        else:
            print(f"FGSM (epsilon={eps}) ===================== MSE on adversarial data: N/A")
    
    print("\nBIM Attack Results:")
    for eps in epsilons:
        key = f"BIM_{eps}"
        mse = results.get(key)
        if mse is not None:
            print(f"BIM (epsilon={eps}) ===================== MSE on adversarial data: {mse:.4f}")
        else:
            print(f"BIM (epsilon={eps}) ===================== MSE on adversarial data: N/A")
            
        # HSJ attack result (only one value, not per epsilon)
    if "HSJ" in results:
        print("\nHopSkipJump Attack Result:")
        print(f"HSJ ===================== MSE on adversarial data: {results['HSJ']:.4f}")
    else:
        print("\nHopSkipJump Attack Result:")
        print("HSJ ===================== MSE on adversarial data: N/A")

    logging.info("======================== Regression Adversarial Scan Playbook completed successfully.")
    
    
if __name__ == "__main__":
    main()
    