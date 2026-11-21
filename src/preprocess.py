"""
Bitcoin Data Preprocessing and Feature Engineering.

Cleans historical BTC-USD data and calculates:
- 7-day moving average
- 30-day moving average
- Daily returns
- 7-day rolling volatility
- 30-day rolling volatility

Saves the processed dataset in data/processed/.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "btc_usd_daily.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "btc_usd_processed.csv"


# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def preprocess_bitcoin_data():

    try:
        logger.info("Loading raw Bitcoin dataset...")

        # Check whether raw dataset exists
        if not RAW_FILE.exists():
            raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

        # Load CSV
        data = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True)

        # Sort records chronologically
        data.sort_index(inplace=True)

        # Remove duplicate dates
        data = data[~data.index.duplicated(keep="last")]

        # Select required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        data = data[required_columns].copy()

        # Convert columns to numeric
        for column in required_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # Handle missing and invalid values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # Remove invalid prices and volume
        data = data[
            (data["Open"] > 0)
            & (data["High"] > 0)
            & (data["Low"] > 0)
            & (data["Close"] > 0)
            & (data["Volume"] >= 0)
        ]

        if len(data) < 31:
            raise ValueError("Insufficient records for feature engineering.")

        logger.info("Calculating technical indicators...")

        # 7-day and 30-day moving averages
        data["MA_7"] = data["Close"].rolling(window=7).mean()

        data["MA_30"] = data["Close"].rolling(window=30).mean()

        # Daily percentage returns
        data["Daily_Return"] = data["Close"].pct_change()

        # Rolling volatility
        data["Volatility_7"] = data["Daily_Return"].rolling(window=7).std()

        data["Volatility_30"] = data["Daily_Return"].rolling(window=30).std()

        # Remove rows where rolling indicators are unavailable
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if data.empty:
            raise ValueError("No records available after feature engineering.")

        # Create output directory
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Save processed data
        data.to_csv(OUTPUT_FILE)

        logger.info("Preprocessing completed successfully!")
        logger.info("Total processed records: %s", len(data))

        print("\nProcessed dataset - First five records:")
        print(data.head())

        print("\nProcessed columns:")
        print(data.columns.tolist())

        print(f"\nTotal processed records: {len(data)}")

        print(f"\nSaved at: {OUTPUT_FILE}")

        return data

    except Exception:
        logger.exception("Preprocessing failed.")
        raise


if __name__ == "__main__":
    preprocess_bitcoin_data()
