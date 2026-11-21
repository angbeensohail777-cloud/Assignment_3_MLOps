
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "btc_usd_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    """Calculate MAE, RMSE and MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    actual = np.array(actual)
    predicted = np.array(predicted)

    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def main():
    # Load processed dataset
    df = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )

    df = df.sort_index()
    prices = df["Close"].astype(float).dropna()

    if len(prices) < 40:
        raise ValueError("Not enough data to train the forecasting model.")

    # Use the latest 7 days as test data
    train_data = prices.iloc[:-7]
    test_data = prices.iloc[-7:]

    print("Training ARIMA model for evaluation...")

    # Train model on historical training data
    evaluation_model = ARIMA(
        train_data,
        order=(5, 1, 0)
    ).fit()

    # Forecast the next 7 days
    predictions = evaluation_model.forecast(steps=7)

    # Evaluate predictions
    metrics = calculate_metrics(
        test_data.values,
        predictions.values
    )

    print("\nModel Evaluation Results")
    print("------------------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Train final model using the complete dataset
    print("\nTraining final model on complete dataset...")

    final_model = ARIMA(
        prices,
        order=(5, 1, 0)
    ).fit()

    # Save trained model
    model_path = MODEL_DIR / "btc_arima_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(final_model, file)

    # Save evaluation metrics
    metrics_path = MODEL_DIR / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Model saved at: {model_path}")
    print(f"Metrics saved at: {metrics_path}")


if __name__ == "__main__":
    main()
    