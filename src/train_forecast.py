
from pathlib import Path
import json
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "btc_usd_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "btc_arima_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"

# Project-defined acceptance threshold
MAPE_THRESHOLD = 10.0


def calculate_metrics(actual, predicted):
    """Calculate MAE, RMSE and MAPE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero_mask = actual != 0

    if not np.any(nonzero_mask):
        raise ValueError("MAPE cannot be calculated because actual values are zero.")

    mape = np.mean(
        np.abs(
            (actual[nonzero_mask] - predicted[nonzero_mask])
            / actual[nonzero_mask]
        )
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def main():
    # Load processed dataset
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    prices = df["Close"].astype(float).dropna()

    if len(prices) < 40:
        raise ValueError("Not enough data to train the forecasting model.")

    # Use the latest 7 days as test data
    train_data = prices.iloc[:-7]
    test_data = prices.iloc[-7:]

    print("Training ARIMA model for evaluation...")

    evaluation_model = ARIMA(
        train_data,
        order=(5, 1, 0)
    ).fit()

    predictions = evaluation_model.forecast(steps=7)

    metrics = calculate_metrics(
        test_data.values,
        predictions.values
    )

    passed = metrics["MAPE"] <= MAPE_THRESHOLD

    print("\nModel Evaluation Results")
    print("------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"MAPE threshold: {MAPE_THRESHOLD:.2f}%")
    print(f"Threshold passed: {passed}")

    # Save evaluation results whether model passes or fails
    metrics_report = {
        **metrics,
        "mape_threshold": MAPE_THRESHOLD,
        "threshold_passed": passed,
        "test_window_days": 7,
        "test_start_date": str(test_data.index[0].date()),
        "test_end_date": str(test_data.index[-1].date()),
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics_report, file, indent=4)

    if not passed:
        print(
            "\nModel did not pass the threshold. "
            "The previously registered model will not be replaced."
        )
        return

    # Train final model using the complete dataset
    print("\nTraining final model on complete dataset...")

    final_model = ARIMA(
        prices,
        order=(5, 1, 0)
    ).fit()

    # Save model to a temporary file first
    temporary_model_path = MODEL_DIR / "btc_arima_model_temp.pkl"

    with open(temporary_model_path, "wb") as file:
        pickle.dump(final_model, file)

    # Replace registered model only after successful training and saving
    os.replace(temporary_model_path, MODEL_PATH)

    # Register model metadata
    registry = {
        "model_name": "BTC ARIMA Forecast Model",
        "model_version": "1.0.0",
        "model_path": str(MODEL_PATH),
        "algorithm": "ARIMA(5,1,0)",
        "registered": True,
        "metrics": metrics,
        "mape_threshold": MAPE_THRESHOLD,
    }

    with open(REGISTRY_PATH, "w") as file:
        json.dump(registry, file, indent=4)

    print("\nModel passed evaluation and was registered successfully!")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Metrics saved at: {METRICS_PATH}")
    print(f"Registry saved at: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
