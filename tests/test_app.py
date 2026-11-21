
import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "model_version" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.get_json()

    assert "MAE" in data["metrics"]
    assert "RMSE" in data["metrics"]
    assert "MAPE" in data["metrics"]


def test_default_prediction(client):
    response = client.post("/predict", json={})

    assert response.status_code == 200

    data = response.get_json()

    assert data["forecast_days"] == 7
    assert len(data["predictions"]) == 7


def test_custom_forecast_days(client):
    response = client.post(
        "/predict",
        json={"forecast_days": 3}
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["forecast_days"] == 3
    assert len(data["predictions"]) == 3


def test_invalid_forecast_days(client):
    response = client.post(
        "/predict",
        json={"forecast_days": 0}
    )

    assert response.status_code == 400


def test_null_request_payload(client):
    response = client.post("/predict", json=None)

    assert response.status_code == 400
