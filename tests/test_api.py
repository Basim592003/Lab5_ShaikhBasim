import pytest
from fastapi.testclient import TestClient

from src.data import CLASS_NAMES, FEATURES, load_data
from src.main import EXAMPLE, WineSample, app


@pytest.fixture(scope="module")
def client():
    # The `with` block runs the app's startup code, which loads the model
    with TestClient(app) as client:
        yield client


def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["classes"] == CLASS_NAMES
    assert info["features"] == FEATURES
    assert 0 <= info["test_accuracy"] <= 1


def test_predict_example(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "cultivar_1"
    assert result["probability"] == max(result["probabilities"].values())
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=0.001)


def test_batch_predicts_each_cultivar(client):
    # The average wine of each cultivar should be classified as that cultivar
    X, y = load_data()
    averages = [X[y == name].mean().to_dict() for name in CLASS_NAMES]
    response = client.post("/predict/batch", json={"samples": averages})
    assert response.status_code == 200
    assert [p["label"] for p in response.json()["predictions"]] == CLASS_NAMES


def test_missing_field_is_rejected(client):
    sample = {k: v for k, v in EXAMPLE.items() if k != "proline"}
    assert client.post("/predict", json=sample).status_code == 422


def test_out_of_range_value_is_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "alcohol": -1}).status_code == 422


def test_unknown_field_is_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "colour": 5}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422


def test_every_wine_in_the_dataset_passes_validation():
    # Checks the limits in WineSample aren't too tight for real wines
    X, _ = load_data()
    for row in X.to_dict("records"):
        WineSample(**row)
