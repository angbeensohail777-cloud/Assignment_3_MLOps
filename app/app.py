
from pathlib import Path
from datetime import datetime, timezone
import pickle
import time
import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from statsmodels.tsa.arima.model import ARIMA


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "btc_arima_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

MODEL_VERSION = "1.0.0"
START_TIME = time.time()

# Load the trained model when the API starts
try:
    with open(MODEL_PATH, "rb") as file:
        trained_model = pickle.load(file)
    print("Trained Bitcoin forecasting model loaded successfully.")
except Exception as error:
    trained_model = None
    print(f"Model loading failed: {error}")


# --------------------------------------------------
# Helper function: create forecast response
# --------------------------------------------------

def generate_forecast(model, forecast_days, last_date):
    predictions = model.forecast(steps=forecast_days)

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    results = []

    for date, price in zip(future_dates, predictions):
        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "predicted_price": round(float(price), 2)
        })

    return results


# --------------------------------------------------
# GET /health
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if trained_model is not None else "unhealthy",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200 if trained_model is not None else 500


# --------------------------------------------------
# POST /predict
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({
                "error": "Request must contain JSON data."
            }), 400

        data = request.get_json()
        # Validate that the request contains a JSON object
        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a valid, non-null JSON object."
            }), 400

        forecast_days = data.get("forecast_days", 7)

        if (
            isinstance(forecast_days, bool)
            or not isinstance(forecast_days, int)
            or forecast_days < 1
            or forecast_days > 30
        ):
            return jsonify({
                "error": "forecast_days must be an integer between 1 and 30."
            }), 400

        if "historical_data" in data:
            historical_data = data["historical_data"]

            if not isinstance(historical_data, list):
                return jsonify({
                    "error": "historical_data must be a list of prices."
                }), 400

            if len(historical_data) < 40:
                return jsonify({
                    "error": "At least 40 historical prices are required."
                }), 400

            try:
                prices = np.array(historical_data, dtype=float)
            except (ValueError, TypeError):
                return jsonify({
                    "error": "Historical prices must be numeric."
                }), 400

            if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
                return jsonify({
                    "error": "Historical prices must be finite positive numbers."
                }), 400

            prediction_model = ARIMA(
                prices,
                order=(5, 1, 0)
            ).fit()

            last_date = pd.Timestamp.today().normalize()

        else:
            if trained_model is None:
                return jsonify({
                    "error": "Trained model is not available."
                }), 500

            prediction_model = trained_model

            date_index = prediction_model.model.data.row_labels

            if date_index is not None and len(date_index) > 0:
                last_date = pd.Timestamp(date_index[-1])
            else:
                last_date = pd.Timestamp.today().normalize()

        forecasts = generate_forecast(
            prediction_model,
            forecast_days,
            last_date
        )

        return jsonify({
            "model_version": MODEL_VERSION,
            "forecast_days": forecast_days,
            "predictions": forecasts
        }), 200

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed.",
            "details": str(error)
        }), 500


# --------------------------------------------------
# GET /metrics
# --------------------------------------------------

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        if not METRICS_PATH.exists():
            return jsonify({
                "error": "Model evaluation metrics are not available."
            }), 500

        with open(METRICS_PATH, "r") as file:
            evaluation_metrics = json.load(file)

        return jsonify({
            "model_version": MODEL_VERSION,
            "metrics": evaluation_metrics
        }), 200

    except Exception as error:
        app.logger.exception("Could not retrieve metrics")
        return jsonify({
            "error": "Could not retrieve model metrics.",
            "details": str(error)
        }), 500


# --------------------------------------------------
# Run application locally
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
