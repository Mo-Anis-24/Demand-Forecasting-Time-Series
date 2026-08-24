import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.exception.exception import forcast
from src.logger import logger


# Project root = two levels up from this file
# src/components/data_ingestion.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    raw_data_path: str = str(PROJECT_ROOT / "data" / "raw")
    clean_output_path: str = str(PROJECT_ROOT / "artifacts" / "data_ingestion" / "orders_clean.parquet")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logger.logging.info("Data ingestion started")

            raw = self.config.raw_data_path

            orders     = pd.read_csv(f"{raw}/orders.csv",     encoding="latin1")
            stores     = pd.read_csv(f"{raw}/stores.csv",     encoding="latin1")
            hubs       = pd.read_csv(f"{raw}/hubs.csv",       encoding="latin1")
            drivers    = pd.read_csv(f"{raw}/drivers.csv",    encoding="latin1")
            deliveries = pd.read_csv(f"{raw}/deliveries.csv", encoding="latin1")
            payments   = pd.read_csv(f"{raw}/payments.csv",   encoding="latin1")
            channels   = pd.read_csv(f"{raw}/channels.csv",   encoding="latin1")

            logger.logging.info(
                f"Loaded raw files — orders: {orders.shape}, stores: {stores.shape}, "
                f"hubs: {hubs.shape}, drivers: {drivers.shape}, deliveries: {deliveries.shape}, "
                f"payments: {payments.shape}, channels: {channels.shape}"
            )

            stores     = stores.merge(hubs,    on="hub_id",    how="left")
            deliveries = deliveries.merge(drivers, on="driver_id", how="left")

            df = (orders
                  .merge(stores,   on="store_id",   how="left")
                  .merge(channels, on="channel_id", how="left"))

            assert len(df) == len(orders), "Merge inflated row count"
            assert df["order_id"].is_unique, "Duplicate order_id found"

            logger.logging.info(f"Merged table shape: {df.shape}")

            df["order_moment_created"] = pd.to_datetime(df["order_moment_created"])

            keep_cols = [
                "order_id", "store_id", "hub_id", "channel_id",
                "order_status", "order_amount",
                "order_moment_created",
                "hub_name", "hub_city", "hub_state",
                "store_segment", "channel_name", "channel_type",
            ]
            df = df[keep_cols]

            logger.logging.info(
                f"Kept {len(keep_cols)} forecast-relevant columns — final shape: {df.shape}"
            )

            os.makedirs(os.path.dirname(self.config.clean_output_path), exist_ok=True)
            df.to_parquet(self.config.clean_output_path, index=False)

            logger.logging.info(f"Saved cleaned data to {self.config.clean_output_path}")

            return self.config.clean_output_path

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    output_path = ingestion.initiate_data_ingestion()
    print(f"Data ingestion complete. Output: {output_path}")