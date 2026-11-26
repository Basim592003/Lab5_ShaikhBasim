import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASS_NAMES, load_data, split_data

# Absolute paths, so training and the API work from any folder
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def fit_model(X_train, y_train):
    """
    Train the classifier. The measurements are on very different scales (hue is around 1,
    proline is in the hundreds), so they are standardised first. The scaler is saved as part
    of the model, so the API can take the raw measurements.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(X_train, y_train)


def evaluate(model, X_test, y_test):
    """Score the model on the test wines it never saw during training."""
    predictions = model.predict(X_test)
    return {
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        # Rows are the true cultivar, columns the predicted one, both in CLASS_NAMES order
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_NAMES).tolist(),
    }


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)

    metrics = {
        "model": "StandardScaler + LogisticRegression",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **evaluate(model, X_test, y_test),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Test accuracy: {metrics['test_accuracy']} on {metrics['test_rows']} wines")
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name} to {MODEL_DIR}")
