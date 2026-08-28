import os
import sys
import pickle
import yaml
from pathlib import Path
import pandas as pd
import mlflow
import dagshub

from src.exception.exception import forcast
from src.logger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.logging.info(f"Object saved: {file_path}")
    except Exception as e:
        raise forcast(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise forcast(e, sys)


def load_yaml(file_path):
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise forcast(e, sys)


def save_parquet(df: pd.DataFrame, file_path: str):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_parquet(file_path, index=False)
        logger.logging.info(f"Parquet saved: {file_path} — shape {df.shape}")
    except Exception as e:
        raise forcast(e, sys)


def setup_mlflow(experiment_name: str = "delivery-hourly-forecast"):
    try:
        dagshub.init(
            repo_owner="manismansuri24",
            repo_name="Demand-Forecasting-Time-Series",
            mlflow=True,
        )
        mlflow.set_experiment(experiment_name)
        logger.logging.info(f"MLflow tracking to DagsHub — experiment: {experiment_name}")
    except Exception as e:
        raise forcast(e, sys)