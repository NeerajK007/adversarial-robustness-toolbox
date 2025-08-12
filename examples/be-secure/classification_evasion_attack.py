import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod, HopSkipJump
from art.attacks.evasion import DeepFool


def create_classification_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Load and preprocess data (using MNIST from TensorFlow)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    num_classes = 10
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    # 2. Build and train model
    input_shape = X_train.shape[1:]
    model = create_classification_model(input_shape, num_classes)
    model.fit(X_train, y_train_oh, epochs=3, batch_size=128, verbose=1)

    # 3. Wrap with ART classifier
    classifier = KerasClassifier(model=model, clip_values=(0, 1))

    # 4. Evaluate on clean data
    preds = np.argmax(classifier.predict(X_test), axis=1)
    acc_clean = np.mean(preds == y_test)
    print(f"Accuracy on clean data: {acc_clean:.4f}")

    # 5. Run attacks
    epsilons = [0.01, 0.05, 0.1, 0.2]
    for eps in epsilons:
        # FGSM
        attack_fgsm = FastGradientMethod(estimator=classifier, eps=eps)
        X_adv_fgsm = attack_fgsm.generate(x=X_test)
        preds_adv = np.argmax(classifier.predict(X_adv_fgsm), axis=1)
        acc_adv = np.mean(preds_adv == y_test)
        print(f"FGSM (epsilon={eps}) accuracy: {acc_adv:.4f}")

        # PGD
        attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=eps)
        X_adv_pgd = attack_pgd.generate(x=X_test)
        preds_adv = np.argmax(classifier.predict(X_adv_pgd), axis=1)
        acc_adv = np.mean(preds_adv == y_test)
        print(f"PGD (epsilon={eps}) accuracy: {acc_adv:.4f}")

        # BIM
        attack_bim = BasicIterativeMethod(estimator=classifier, eps=eps)
        X_adv_bim = attack_bim.generate(x=X_test)
        preds_adv = np.argmax(classifier.predict(X_adv_bim), axis=1)
        acc_adv = np.mean(preds_adv == y_test)
        print(f"BIM (epsilon={eps}) accuracy: {acc_adv:.4f}")

    # 6. HopSkipJump (decision-based, no epsilon)
    attack_hsj = HopSkipJump(classifier=classifier, max_iter=10)
    X_adv_hsj = attack_hsj.generate(x=X_test)
    preds_adv = np.argmax(classifier.predict(X_adv_hsj), axis=1)
    acc_adv = np.mean(preds_adv == y_test)
    print(f"HopSkipJump accuracy: {acc_adv:.4f}")
    
    # DeepFool (no epsilon parameter)
    attack_deepfool = DeepFool(classifier=classifier)
    X_adv_deepfool = attack_deepfool.generate(x=X_test)
    preds_adv = np.argmax(classifier.predict(X_adv_deepfool), axis=1)
    acc_adv = np.mean(preds_adv == y_test)
    print(f"DeepFool accuracy: {acc_adv:.4f}")

if __name__ == "__main__":
    main()