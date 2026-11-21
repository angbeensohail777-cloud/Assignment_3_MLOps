
from datetime import datetime, timedelta
import os

import requests

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


# Flask API address inside Docker Compose network.
# We can change this if your Flask service has a different name.
FLASK_API_URL = os.getenv(
    "FLASK_API_URL",
    "http://flask-api:5000"
)


def notify_api_reload():
    """Notify Flask API to reload the newly trained model."""

    response = requests.post(
        f"{FLASK_API_URL}/reload",
        timeout=30
    )

    response.raise_for_status()

    print("Flask API model reload notification sent successfully.")


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="btc_etl_pipeline",
    default_args=default_args,
    description="Daily BTC data ingestion, preprocessing and model retraining",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["bitcoin", "mlops", "forecasting"],
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_btc_data",
        bash_command="python /opt/airflow/src/etl_fetch.py",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_and_engineer_features",
        bash_command="python /opt/airflow/src/preprocess.py",
    )

    train_model = BashOperator(
        task_id="train_and_evaluate_model",
        bash_command="python /opt/airflow/src/train_forecast.py",
    )

    reload_api_model = PythonOperator(
        task_id="notify_api_model_reload",
        python_callable=notify_api_reload,
    )

    fetch_data >> preprocess_data >> train_model >> reload_api_model
