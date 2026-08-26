import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.exception.exception import forcast
from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelTrainerConfig:
    """
    Paths, split strategy, and model hyperparameters.
    """
    input_path: str        = str(PROJECT_ROOT / "artifacts" / "data_transformation" / "features.parquet")
    model_path: str        = str(PROJECT_ROOT / "artifacts" / "model_trainer" / "lgbm_model.pkl")
    metrics_path: str      = str(PROJECT_ROOT / "artifacts" / "model_trainer" / "metrics.txt")
    predictions_path: str  = str(PROJECT_ROOT / "artifacts" / "model_trainer" / "test_predictions.csv")
    importance_path: str   = str(PROJECT_ROOT / "artifacts" / "model_trainer" / "feature_importance.csv")

    test_days: int         = 14
    target_col: str        = "orders"
    exclude_cols: List[str] = field(default_factory=lambda: [
        "ts", "hub_name", "hub_total_orders", "orders"
    ])
    cat_features: List[str] = field(default_factory=lambda: [
        "hub_id", "hub_city", "hub_state", "hub_dominant_segment", "hub_dominant_channel"
    ])

    # LightGBM hyperparameters
    n_estimators: int      = 3000
    learning_rate: float   = 0.05
    num_leaves: int        = 64
    min_child_samples: int = 20
    subsample: float       = 0.8
    colsample_bytree: float = 0.8
    objective: str         = "poisson"
    random_state: int      = 42
    early_stopping_rounds: int = 100


class ModelTrainer:
    """
    Trains a LightGBM model to forecast hourly hub-level order demand.

    Steps:
      1. Load engineered features
      2. Time-based train/test split (last N days = test)
      3. Compute two baselines (seasonal-naive, rolling mean)
      4. Train LightGBM with Poisson objective + early stopping
      5. Evaluate on test set (MAE, RMSE, WAPE)
      6. Save model, metrics, predictions, feature importance
    """

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        try:
            logger.logging.info("Model training started")

            # ---- 1. Load features ----
            df = pd.read_parquet(self.config.input_path)
            logger.logging.info(
                f"Loaded features: {df.shape}, hubs: {df['hub_id'].nunique()}, "
                f"range: {df['ts'].min()} → {df['ts'].max()}"
            )

            # ---- 2. Time-based split ----
            cutoff = df["ts"].max() - pd.Timedelta(days=self.config.test_days)
            train = df[df["ts"] <= cutoff].copy()
            test  = df[df["ts"] >  cutoff].copy()
            logger.logging.info(
                f"Train rows: {len(train)}, Test rows: {len(test)} "
                f"(cutoff: {cutoff})"
            )

            # ---- 3. Feature/target separation ----
            features = [c for c in df.columns if c not in self.config.exclude_cols]
            X_train, y_train = train[features], train[self.config.target_col]
            X_test,  y_test  = test[features],  test[self.config.target_col]
            logger.logging.info(f"Using {len(features)} features")

            # ---- 4. Baseline #1: seasonal naive (lag_168) ----
            baseline_pred = X_test["lag_168"].values
            baseline_mae  = mean_absolute_error(y_test, baseline_pred)
            baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
            baseline_wape = np.abs(y_test - baseline_pred).sum() / y_test.sum()
            logger.logging.info(
                f"Baseline (seasonal naive) — MAE: {baseline_mae:.3f}, "
                f"RMSE: {baseline_rmse:.3f}, WAPE: {baseline_wape*100:.2f}%"
            )

            # ---- 5. Baseline #2: 7-day rolling mean ----
            baseline2_pred = X_test["roll_mean_168"].values
            baseline2_mae  = mean_absolute_error(y_test, baseline2_pred)
            baseline2_rmse = np.sqrt(mean_squared_error(y_test, baseline2_pred))
            baseline2_wape = np.abs(y_test - baseline2_pred).sum() / y_test.sum()
            logger.logging.info(
                f"Baseline (rolling mean 168h) — MAE: {baseline2_mae:.3f}, "
                f"WAPE: {baseline2_wape*100:.2f}%"
            )

            # ---- 6. Train LightGBM ----
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                objective=self.config.objective,
                random_state=self.config.random_state,
                verbose=-1,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                categorical_feature=self.config.cat_features,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(200),
                ],
            )
            logger.logging.info(f"Model trained — best iteration: {model.best_iteration_}")

            # ---- 7. Evaluate ----
            pred = np.clip(model.predict(X_test), 0, None)
            lgbm_mae  = mean_absolute_error(y_test, pred)
            lgbm_rmse = np.sqrt(mean_squared_error(y_test, pred))
            lgbm_wape = np.abs(y_test - pred).sum() / y_test.sum()

            improvement_mae  = (1 - lgbm_mae  / baseline_mae)  * 100
            improvement_wape = (1 - lgbm_wape / baseline_wape) * 100

            logger.logging.info(
                f"LightGBM — MAE: {lgbm_mae:.3f}, RMSE: {lgbm_rmse:.3f}, "
                f"WAPE: {lgbm_wape*100:.2f}%"
            )
            logger.logging.info(
                f"Improvement over seasonal-naive — "
                f"MAE: {improvement_mae:.1f}%, WAPE: {improvement_wape:.1f}%"
            )

            # ---- 8. Save model ----
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            with open(self.config.model_path, "wb") as f:
                pickle.dump(model, f)
            logger.logging.info(f"Model saved to {self.config.model_path}")

            # ---- 9. Save metrics summary ----
            metrics = {
                "test_period_days":       self.config.test_days,
                "train_rows":             len(train),
                "test_rows":              len(test),
                "n_features":             len(features),
                "best_iteration":         model.best_iteration_,
                "baseline_seasonal_mae":  round(baseline_mae, 4),
                "baseline_seasonal_wape": round(baseline_wape * 100, 2),
                "baseline_rolling_mae":   round(baseline2_mae, 4),
                "baseline_rolling_wape":  round(baseline2_wape * 100, 2),
                "lgbm_mae":               round(lgbm_mae, 4),
                "lgbm_rmse":              round(lgbm_rmse, 4),
                "lgbm_wape":              round(lgbm_wape * 100, 2),
                "improvement_mae_pct":    round(improvement_mae, 2),
                "improvement_wape_pct":   round(improvement_wape, 2),
            }
            with open(self.config.metrics_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
            logger.logging.info(f"Metrics saved to {self.config.metrics_path}")

            # ---- 10. Save test predictions (for downstream analysis / dashboard) ----
            test_eval = test[["hub_id", "hub_name", "ts", "orders"]].copy()
            test_eval["pred"] = pred
            test_eval.to_csv(self.config.predictions_path, index=False)
            logger.logging.info(f"Predictions saved to {self.config.predictions_path}")

            # ---- 11. Save feature importance ----
            imp = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp.to_csv(self.config.importance_path, index=False)
            logger.logging.info(
                f"Feature importance saved — top 5: {imp.head(5)['feature'].tolist()}"
            )

            return metrics

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer()
    print("\nModel training complete. Key metrics:")
    for k in ["lgbm_mae", "lgbm_wape", "improvement_mae_pct"]:
        print(f"  {k:24s}: {metrics[k]}")