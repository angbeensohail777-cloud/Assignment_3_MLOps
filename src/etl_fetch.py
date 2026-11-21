
"""
Bitcoin Historical Data Fetcher
Downloads daily BTC-USD data from Yahoo Finance
and saves the raw dataset in data/raw/.
"""

from pathlib import Path
import logging

import yfinance as yf


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data storage location
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Output CSV file
OUTPUT_FILE = RAW_DATA_DIR / "btc_usd_daily.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def fetch_bitcoin_data():
    """Download historical daily Bitcoin market data."""

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading BTC-USD historical data...")

        data = yf.download(
            tickers="BTC-USD",
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError("Yahoo Finance returned an empty dataset.")

        # Handle possible multi-level column names
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Keep required financial columns
        data = data[required_columns].copy()

        # Remove rows with missing values
        data.dropna(inplace=True)

        if data.empty:
            raise ValueError("No valid records remain after cleaning.")

        # Save raw dataset
        data.to_csv(OUTPUT_FILE)

        logger.info("Bitcoin data downloaded successfully!")
        logger.info("Total records: %s", len(data))
        logger.info("Saved dataset: %s", OUTPUT_FILE)

        print("\nFirst five records:")
        print(data.head())

        print("\nLast five records:")
        print(data.tail())

        print(f"\nTotal records: {len(data)}")
        print(f"Dataset saved at: {OUTPUT_FILE}")

        return data

    except Exception:
        logger.exception("Bitcoin data fetching failed.")
        raise


if __name__ == "__main__":
    fetch_bitcoin_data()


