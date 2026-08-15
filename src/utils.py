import os
import sys
import pickle
import yaml
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Save any Python object to disk as a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a previously saved pickle object."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_yaml(file_path):
    """Load a YAML config file into a dict."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)


def save_parquet(df: pd.DataFrame, file_path: str):
    """Save a DataFrame to parquet format."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_parquet(file_path, index=False)
        logging.info(f"Saved parquet to {file_path} — shape {df.shape}")
    except Exception as e:
        raise CustomException(e, sys)