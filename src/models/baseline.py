import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import pickle
import os

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
os.makedirs(MODELS_PATH, exist_ok=True)


def load_data():
    X_train = np.load(f"{PROCESSED_PATH}/X_train.npy")
    X_test  = np.load(f"{PROCESSED_PATH}/X_test.npy")
    y_train = np.load(f"{PROCESSED_PATH}/y_train.npy")
    y_test  = np.load(f"{PROCESSED_PATH}/y_test.npy")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["Control", "Alzheimer's"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


def run_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.01,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def run_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, name):
    path = f"{MODELS_PATH}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    lr = run_logistic_regression(X_train, y_train)
    evaluate("Logistic Regression", lr, X_test, y_test)
    save_model(lr, "logistic_regression")

    rf = run_random_forest(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)
    save_model(rf, "random_forest")