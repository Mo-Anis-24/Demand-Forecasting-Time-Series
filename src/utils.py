import os
import sys
import pickle
import yaml
from pathlib import Path
import pandas as pd
import mlflow

from src.exception.exception import forcast
from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def save_object(file_path, obj):
    """Save any Python object to disk as a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise forcast(e, sys)


def load_object(file_path):
    """Load a previously saved pickle object."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise forcast(e, sys)


def load_yaml(file_path):
    """Load a YAML config file into a dict."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise forcast(e, sys)


def save_parquet(df: pd.DataFrame, file_path: str):
    """Save a DataFrame to parquet format."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_parquet(file_path, index=False)
        logger.logging.info(f"Saved parquet to {file_path} — shape {df.shape}")
    except Exception as e:
        raise forcast(e, sys)


def setup_mlflow(experiment_name: str = "delivery-hourly-forecast"):
    """
    Configure MLflow to store runs locally under ./mlruns and set the experiment.
    Call this once at the start of any component that logs to MLflow.
    """
    try:
        MLRUNS_DIR.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(experiment_name)
        logger.logging.info(f"MLflow configured — experiment: {experiment_name}")
    except Exception as e:
        raise forcast(e, sys)