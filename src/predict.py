import json

import joblib
import pandas as pd

from .data import FEATURES
from .train import METRICS_PATH, MODEL_PATH


def load_model():
    """Load the trained model, with a clear message if it hasn't been trained yet."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found; run `python -m src.train` first")
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load the test results saved next to the model by train.py."""
    return json.loads(METRICS_PATH.read_text())


def predict_data(model, samples):
    """
    Predict the cultivar of one or more wines.
    Args:
        model: The trained model from load_model().
        samples (list[dict]): One dict of the 13 measurements per wine.
    Returns:
        list[dict]: For each wine, the most likely cultivar, its probability, and the
        probability of every cultivar.
    """
    X = pd.DataFrame(samples, columns=FEATURES)
    results = []
    for row in model.predict_proba(X):
        best = row.argmax()
        results.append({
            "label": str(model.classes_[best]),
            "probability": round(float(row[best]), 4),
            "probabilities": {str(label): round(float(p), 4) for label, p in zip(model.classes_, row)},
        })
    return results
