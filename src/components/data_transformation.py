import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

from src.exception.exception import forcast
from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataTransformationConfig:
    """
    Paths and thresholds for feature engineering.
    """
    input_path: str = str(PROJECT_ROOT / "artifacts" / "data_ingestion" / "orders_clean.parquet")
    output_path: str = str(PROJECT_ROOT / "artifacts" / "data_transformation" / "features.parquet")
    min_orders_per_hub: int = 500   # drop hubs with less than this much activity


class DataTransformation:
    """
    Turns cleaned order-level data into a (hub × hour) feature table
    ready for modeling.

    Steps:
      1. Filter to FINISHED orders
      2. Aggregate to hub-hour level
      3. Fill missing hub-hours with zero (rectangular grid)
      4. Drop inactive hubs
      5. Attach hub metadata (name, city, state, dominant segment/channel)
      6. Calendar + cyclical + part-of-day + holiday + payday features
      7. Lag features (1h, 24h, 168h, 336h)
      8. Rolling stats (mean, std, max)
      9. Hub-level statics (total orders, avg hourly)
     10. Drop rows lacking history (from lag_336 requirement)
     11. Cast categoricals, save to parquet
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            logger.logging.info("Data transformation started")

            # ---- 1. Load cleaned data ----
            df = pd.read_parquet(self.config.input_path)
            logger.logging.info(f"Loaded input: {df.shape}")

            # ---- 2. Filter to FINISHED orders ----
            df = df[df["order_status"] == "FINISHED"].copy()
            logger.logging.info(f"Rows after FINISHED filter: {len(df)}")

            # ---- 3. Floor timestamp to the hour ----
            df["order_moment_created"] = pd.to_datetime(df["order_moment_created"])
            df["ts"] = df["order_moment_created"].dt.floor("H")

            # ---- 4. Aggregate to hub-hour level ----
            hourly = (df.groupby(["hub_id", "ts"])["order_id"]
                        .nunique()
                        .reset_index(name="orders"))
            logger.logging.info(f"Aggregated shape (before zero-fill): {hourly.shape}")

            # ---- 5. Fill missing hub-hours with zeros ----
            all_hubs = hourly["hub_id"].unique()
            all_hours = pd.date_range(hourly["ts"].min(), hourly["ts"].max(), freq="H")
            full_idx = pd.MultiIndex.from_product([all_hubs, all_hours], names=["hub_id", "ts"])
            hourly = (hourly.set_index(["hub_id", "ts"])
                            .reindex(full_idx, fill_value=0)
                            .reset_index())
            hourly = hourly.sort_values(["hub_id", "ts"]).reset_index(drop=True)
            logger.logging.info(
                f"Shape after zero-fill: {hourly.shape}, "
                f"zero-order share: {(hourly['orders']==0).mean()*100:.1f}%"
            )

            # ---- 6. Drop inactive hubs ----
            hub_totals = hourly.groupby("hub_id", observed=True)["orders"].sum()
            active_hubs = hub_totals[hub_totals >= self.config.min_orders_per_hub].index
            before = len(hourly)
            hourly = hourly[hourly["hub_id"].isin(active_hubs)].reset_index(drop=True)
            logger.logging.info(
                f"Dropped {before - len(hourly):,} rows from inactive hubs — "
                f"{hourly['hub_id'].nunique()} hubs kept"
            )

            # ---- 7. Attach hub metadata ----
            hub_info = df[["hub_id", "hub_name", "hub_city", "hub_state"]].drop_duplicates("hub_id")
            hourly = hourly.merge(hub_info, on="hub_id", how="left")

            seg = (df.groupby("hub_id")["store_segment"]
                     .agg(lambda s: s.value_counts().index[0])
                     .reset_index()
                     .rename(columns={"store_segment": "hub_dominant_segment"}))
            hourly = hourly.merge(seg, on="hub_id", how="left")

            ch = (df.groupby("hub_id")["channel_type"]
                    .agg(lambda s: s.value_counts().index[0])
                    .reset_index()
                    .rename(columns={"channel_type": "hub_dominant_channel"}))
            hourly = hourly.merge(ch, on="hub_id", how="left")

            logger.logging.info("Attached hub metadata")

            # ---- 8. Calendar features ----
            hourly["hour"]       = hourly["ts"].dt.hour
            hourly["dow"]        = hourly["ts"].dt.dayofweek
            hourly["day"]        = hourly["ts"].dt.day
            hourly["month"]      = hourly["ts"].dt.month
            hourly["week"]       = hourly["ts"].dt.isocalendar().week.astype(int)
            hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)

            # ---- 9. Cyclical encodings ----
            hourly["hour_sin"] = np.sin(2*np.pi*hourly["hour"]/24)
            hourly["hour_cos"] = np.cos(2*np.pi*hourly["hour"]/24)
            hourly["dow_sin"]  = np.sin(2*np.pi*hourly["dow"]/7)
            hourly["dow_cos"]  = np.cos(2*np.pi*hourly["dow"]/7)

            # ---- 10. Part-of-day flags (evening-heavy platform) ----
            hourly["is_active_window"] = (hourly["hour"].between(13, 23) | (hourly["hour"] == 0)).astype(int)
            hourly["is_dead_window"]   = hourly["hour"].between(2, 12).astype(int)
            hourly["is_evening_peak"]  = hourly["hour"].between(18, 22).astype(int)
            hourly["is_late_night"]    = (hourly["hour"].between(22, 23) | (hourly["hour"] == 0)).astype(int)

            # ---- 11. Holiday + payday flags ----
            br_holidays = holidays.country_holidays("BR", years=[2021, 2022])
            hourly["is_holiday"] = hourly["ts"].dt.date.isin(br_holidays).astype(int)
            hourly["is_payday_window"] = hourly["day"].isin([5, 6, 20, 21]).astype(int)

            logger.logging.info("Calendar, cyclical, and special-day features added")

            # ---- 12. Lag features (per hub) ----
            hourly["lag_1"]   = hourly.groupby("hub_id")["orders"].shift(1)
            hourly["lag_24"]  = hourly.groupby("hub_id")["orders"].shift(24)
            hourly["lag_168"] = hourly.groupby("hub_id")["orders"].shift(168)
            hourly["lag_336"] = hourly.groupby("hub_id")["orders"].shift(336)

            # ---- 13. Rolling stats (per hub, shifted by 1 to avoid leakage) ----
            hourly["roll_mean_24"]  = (hourly.groupby("hub_id")["orders"]
                                             .shift(1).rolling(24).mean()
                                             .reset_index(0, drop=True))
            hourly["roll_mean_168"] = (hourly.groupby("hub_id")["orders"]
                                             .shift(1).rolling(168).mean()
                                             .reset_index(0, drop=True))
            hourly["roll_std_24"]   = (hourly.groupby("hub_id")["orders"]
                                             .shift(1).rolling(24).std()
                                             .reset_index(0, drop=True))
            hourly["roll_std_168"]  = (hourly.groupby("hub_id")["orders"]
                                             .shift(1).rolling(168).std()
                                             .reset_index(0, drop=True))
            hourly["roll_max_24"]   = (hourly.groupby("hub_id")["orders"]
                                             .shift(1).rolling(24).max()
                                             .reset_index(0, drop=True))

            # ---- 14. Same-hour 2-week aggregate ----
            hourly["same_hour_2w_mean"] = hourly[["lag_168", "lag_336"]].mean(axis=1)

            logger.logging.info("Lag and rolling features added")

            # ---- 15. Hub-level statics ----
            hub_totals_df = df.groupby("hub_id").size().reset_index(name="hub_total_orders")
            hub_totals_df["hub_avg_hourly"] = hub_totals_df["hub_total_orders"] / len(all_hours)
            hourly = hourly.merge(hub_totals_df, on="hub_id", how="left")

            # ---- 16. Drop rows lacking history ----
            before = len(hourly)
            hourly = hourly.dropna(subset=["lag_336", "roll_mean_168"]).reset_index(drop=True)
            logger.logging.info(
                f"Dropped {before - len(hourly):,} rows lacking history — "
                f"final shape: {hourly.shape}"
            )

            # ---- 17. Cast categoricals ----
            for col in ["hub_id", "hub_city", "hub_state",
                        "hub_dominant_segment", "hub_dominant_channel"]:
                hourly[col] = hourly[col].astype("category")

            # ---- 18. Leakage validation (fail-fast if features are broken) ----
            sample = hourly[hourly["hub_id"] == hourly["hub_id"].unique()[0]].reset_index(drop=True)
            check_168 = (sample["lag_168"].iloc[168:].values ==
                         sample["orders"].iloc[:-168].values).mean()
            assert check_168 > 0.99, f"lag_168 leakage check failed: {check_168*100:.1f}%"
            logger.logging.info(f"Leakage check passed: lag_168 integrity {check_168*100:.1f}%")

            # ---- 19. Save ----
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
            hourly.to_parquet(self.config.output_path, index=False)
            logger.logging.info(
                f"Saved features to {self.config.output_path} — "
                f"shape {hourly.shape}, hubs {hourly['hub_id'].nunique()}, "
                f"missing values {hourly.isna().sum().sum()}"
            )

            return self.config.output_path

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    output_path = transformer.initiate_data_transformation()
    print(f"Data transformation complete. Output: {output_path}")