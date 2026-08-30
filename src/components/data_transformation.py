import os
import sys
import pickle
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
    input_path: str = str(PROJECT_ROOT / "artifacts" / "data_ingestion" / "orders_clean.parquet")
    output_path: str = str(PROJECT_ROOT / "artifacts" / "data_transformation" / "features.parquet")
    transformation_path: str = str(PROJECT_ROOT / "final_model" / "transformation.pkl")
    min_orders_per_hub: int = 500


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def save_transformation(self, active_hubs, feature_cols, cat_features, hourly, hub_info, seg, ch):
        try:
            os.makedirs(os.path.dirname(self.config.transformation_path), exist_ok=True)

            cat_mappings = {}
            for col in cat_features:
                if hourly[col].dtype.name == "category":
                    cat_mappings[col] = list(hourly[col].cat.categories)

            transformation_state = {
                "active_hubs": list(active_hubs),
                "feature_cols": feature_cols,
                "cat_features": cat_features,
                "cat_mappings": cat_mappings,
                "hub_info": hub_info.to_dict(orient="records"),
                "hub_dominant_segment": seg.set_index("hub_id")["hub_dominant_segment"].to_dict(),
                "hub_dominant_channel": ch.set_index("hub_id")["hub_dominant_channel"].to_dict(),
                "min_orders_per_hub": self.config.min_orders_per_hub,
            }

            with open(self.config.transformation_path, "wb") as f:
                pickle.dump(transformation_state, f)

            logger.logging.info(f"Transformation saved: {self.config.transformation_path}")

        except Exception as e:
            raise forcast(e, sys)

    def initiate_data_transformation(self):
        try:
            logger.logging.info("Data transformation started")

            # Load cleaned data
            df = pd.read_parquet(self.config.input_path)
            df = df[df["order_status"] == "FINISHED"].copy()
            df["order_moment_created"] = pd.to_datetime(df["order_moment_created"])
            df["ts"] = df["order_moment_created"].dt.floor("h")

            # Aggregate to hub-hour
            hourly = (df.groupby(["hub_id", "ts"])["order_id"]
                        .nunique()
                        .reset_index(name="orders"))

            # Fill missing hub-hours with zeros
            all_hubs = hourly["hub_id"].unique()
            all_hours = pd.date_range(hourly["ts"].min(), hourly["ts"].max(), freq="h")
            full_idx = pd.MultiIndex.from_product([all_hubs, all_hours], names=["hub_id", "ts"])
            hourly = (hourly.set_index(["hub_id", "ts"])
                            .reindex(full_idx, fill_value=0)
                            .reset_index())
            hourly = hourly.sort_values(["hub_id", "ts"]).reset_index(drop=True)

            # Drop inactive hubs
            hub_totals = hourly.groupby("hub_id", observed=True)["orders"].sum()
            active_hubs = hub_totals[hub_totals >= self.config.min_orders_per_hub].index
            hourly = hourly[hourly["hub_id"].isin(active_hubs)].reset_index(drop=True)
            logger.logging.info(f"{hourly['hub_id'].nunique()} active hubs kept")

            # Attach hub metadata
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

            # Calendar features
            hourly["hour"]       = hourly["ts"].dt.hour
            hourly["dow"]        = hourly["ts"].dt.dayofweek
            hourly["day"]        = hourly["ts"].dt.day
            hourly["month"]      = hourly["ts"].dt.month
            hourly["week"]       = hourly["ts"].dt.isocalendar().week.astype(int)
            hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)

            # Cyclical encodings
            hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
            hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)
            hourly["dow_sin"]  = np.sin(2 * np.pi * hourly["dow"] / 7)
            hourly["dow_cos"]  = np.cos(2 * np.pi * hourly["dow"] / 7)

            # Part-of-day flags
            hourly["is_active_window"] = (hourly["hour"].between(13, 23) | (hourly["hour"] == 0)).astype(int)
            hourly["is_dead_window"]   = hourly["hour"].between(2, 12).astype(int)
            hourly["is_evening_peak"]  = hourly["hour"].between(18, 22).astype(int)
            hourly["is_late_night"]    = (hourly["hour"].between(22, 23) | (hourly["hour"] == 0)).astype(int)

            # Holiday and payday flags
            br_holidays = holidays.country_holidays("BR", years=[2021, 2022])
            hourly["is_holiday"] = hourly["ts"].dt.date.isin(br_holidays).astype(int)
            hourly["is_payday_window"] = hourly["day"].isin([5, 6, 20, 21]).astype(int)

            # Lag features
            hourly["lag_1"]   = hourly.groupby("hub_id")["orders"].shift(1)
            hourly["lag_24"]  = hourly.groupby("hub_id")["orders"].shift(24)
            hourly["lag_168"] = hourly.groupby("hub_id")["orders"].shift(168)
            hourly["lag_336"] = hourly.groupby("hub_id")["orders"].shift(336)

            # Rolling stats (shifted by 1 to avoid leakage)
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

            # Same-hour 2-week aggregate
            hourly["same_hour_2w_mean"] = hourly[["lag_168", "lag_336"]].mean(axis=1)

            # Hub-level statics
            hub_totals_df = df.groupby("hub_id").size().reset_index(name="hub_total_orders")
            hub_totals_df["hub_avg_hourly"] = hub_totals_df["hub_total_orders"] / len(all_hours)
            hourly = hourly.merge(hub_totals_df, on="hub_id", how="left")

            # Drop rows lacking history
            hourly = hourly.dropna(subset=["lag_336", "roll_mean_168"]).reset_index(drop=True)

            # Cast categoricals
            cat_features = ["hub_id", "hub_city", "hub_state",
                            "hub_dominant_segment", "hub_dominant_channel"]
            for col in cat_features:
                hourly[col] = hourly[col].astype("category")

            # Leakage check
            sample = hourly[hourly["hub_id"] == hourly["hub_id"].unique()[0]].reset_index(drop=True)
            check_168 = (sample["lag_168"].iloc[168:].values ==
                         sample["orders"].iloc[:-168].values).mean()
            assert check_168 > 0.99, f"lag_168 leakage check failed: {check_168*100:.1f}%"

            # Save features parquet
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
            hourly.to_parquet(self.config.output_path, index=False)
            logger.logging.info(f"Features saved: {self.config.output_path} — shape {hourly.shape}")

            # Save transformation.pkl to final_model/
            exclude_cols = ["ts", "hub_name", "hub_total_orders", "orders"]
            feature_cols = [c for c in hourly.columns if c not in exclude_cols]
            self.save_transformation(active_hubs, feature_cols, cat_features, hourly, hub_info, seg, ch)

            return self.config.output_path

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    output_path = transformer.initiate_data_transformation()
    print(f"Data transformation complete. Output: {output_path}")