
from kfp import compiler, dsl
from kfp.dsl import Dataset, Model, Artifact, Input, Output


# Component 1: BTC Data Ingestion
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["yfinance==0.2.66", "pandas==2.2.3"],
)
def ingest_btc_data(output_dataset: Output[Dataset]):
    import pandas as pd
    import yfinance as yf

    data = yf.download(
        "BTC-USD",
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No BTC data was downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.to_csv(output_dataset.path, index=False)

    print(f"Downloaded {len(data)} BTC records.")


# Component 2: Feature Generation
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas==2.2.3"],
)
def generate_btc_features(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
):
    import pandas as pd

    data = pd.read_csv(input_dataset.path)

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"])

    data["MA_7"] = data["Close"].rolling(window=7).mean()
    data["MA_30"] = data["Close"].rolling(window=30).mean()

    data["Daily_Return"] = data["Close"].pct_change()

    data["Volatility_7"] = (
        data["Daily_Return"].rolling(window=7).std()
    )

    data["Volatility_30"] = (
        data["Daily_Return"].rolling(window=30).std()
    )

    data = data.dropna()
    data.to_csv(output_dataset.path, index=False)

    print(f"Generated features for {len(data)} records.")


# Component 3: Model Training and Evaluation
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas==2.2.3",
        "statsmodels==0.14.4",
        "scikit-learn==1.6.1",
    ],
)
def train_btc_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Artifact],
):
    import json
    import pickle

    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    data = pd.read_csv(input_dataset.path)
    prices = data["Close"].dropna().astype(float)

    if len(prices) <= 7:
        raise ValueError("Not enough BTC records for training.")

    train_data = prices.iloc[:-7]
    test_data = prices.iloc[-7:]

    # Train and evaluate using the latest 7 days as test data
    test_model = ARIMA(train_data, order=(5, 1, 0))
    test_model_fit = test_model.fit()

    predictions = test_model_fit.forecast(steps=7)

    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)
    rmse = mse ** 0.5

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "test_days": 7,
    }

    # Train final model using all available data
    final_model = ARIMA(prices, order=(5, 1, 0))
    final_model_fit = final_model.fit()

    with open(output_model.path, "wb") as model_file:
        pickle.dump(final_model_fit, model_file)

    with open(output_metrics.path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print("BTC model training and evaluation completed.")
    print(metrics)


# Define the complete pipeline
@dsl.pipeline(
    name="btc-forecasting-pipeline",
    description=(
        "BTC data ingestion, feature generation, "
        "and ARIMA model training."
    ),
)
def btc_forecasting_pipeline():
    ingestion_task = ingest_btc_data()

    feature_task = generate_btc_features(
        input_dataset=ingestion_task.outputs["output_dataset"]
    )

    training_task = train_btc_model(
        input_dataset=feature_task.outputs["output_dataset"]
    )


# Compile pipeline into YAML
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=btc_forecasting_pipeline,
        package_path="kubeflow/btc_kfp_pipeline.yaml",
    )

    print("Kubeflow pipeline compiled successfully.")
    print("YAML saved to kubeflow/btc_kfp_pipeline.yaml")
