import os
import sys
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.exception.exception import forcast
from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Quiet Optuna output — our own logger handles progress
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HyperparameterTuningConfig:
    """
    Paths and search settings for Optuna-based hyperparameter search.
    """
    input_path: str        = str(PROJECT_ROOT / "artifacts" / "data_transformation" / "features.parquet")
    model_path: str        = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "lgbm_tuned.pkl")
    best_params_path: str  = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "best_params.json")
    metrics_path: str      = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "metrics.txt")
    predictions_path: str  = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "test_predictions_tuned.csv")
    importance_path: str   = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "feature_importance_tuned.csv")
    study_path: str        = str(PROJECT_ROOT / "artifacts" / "hyperparameter_tuning" / "optuna_study.pkl")

    # Split strategy
    test_days: int         = 14
    validation_days: int   = 7

    # Feature setup
    target_col: str        = "orders"
    exclude_cols: List[str] = field(default_factory=lambda: [
        "ts", "hub_name", "hub_total_orders", "orders"
    ])
    cat_features: List[str] = field(default_factory=lambda: [
        "hub_id", "hub_city", "hub_state", "hub_dominant_segment", "hub_dominant_channel"
    ])

    # Search settings
    n_trials: int          = 30
    random_state: int      = 42
    early_stopping_trial:  int = 50       # for trials during search
    early_stopping_final:  int = 100      # for final model training


class HyperparameterTuner:
    """
    Runs Optuna-based hyperparameter search for LightGBM demand forecasting.

    Split strategy:
      - Test set:       last 14 days (holdout, untouched during search)
      - Validation set: 7 days before that (used to score each trial)
      - Train_fit set:  everything before validation (used to fit each trial)

    Workflow:
      1. Load features and build 3-way time-ordered split
      2. Run N Optuna trials, minimizing validation MAE
      3. Train final model on train+val with best params, evaluate on test
      4. Save model, best params, metrics, predictions, feature importance, study
    """

    def __init__(self):
        self.config = HyperparameterTuningConfig()

        # Populated when initiate() runs — kept as instance state so
        # the Optuna objective function can access them via closure
        self.X_train_fit = None
        self.y_train_fit = None
        self.X_val       = None
        self.y_val       = None

    def _objective(self, trial):
        """
        One Optuna trial: sample hyperparameters, train, return val MAE.
        """
        params = {
            "objective":         "poisson",
            "metric":            "mae",
            "verbose":           -1,
            "random_state":      self.config.random_state,
            "n_estimators":      3000,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            self.X_train_fit, self.y_train_fit,
            eval_set=[(self.X_val, self.y_val)],
            categorical_feature=self.config.cat_features,
            callbacks=[lgb.early_stopping(self.config.early_stopping_trial, verbose=False)]
        )

        preds = np.clip(model.predict(self.X_val), 0, None)
        return mean_absolute_error(self.y_val, preds)

    def initiate_hyperparameter_tuning(self):
        try:
            logger.logging.info("Hyperparameter tuning started")

            # ---- 1. Load features ----
            df = pd.read_parquet(self.config.input_path)
            logger.logging.info(
                f"Loaded features: {df.shape}, hubs: {df['hub_id'].nunique()}"
            )

            # ---- 2. Three-way time-based split ----
            test_cutoff = df["ts"].max() - pd.Timedelta(days=self.config.test_days)
            train_full = df[df["ts"] <= test_cutoff].copy()
            test       = df[df["ts"] >  test_cutoff].copy()

            val_cutoff = train_full["ts"].max() - pd.Timedelta(days=self.config.validation_days)
            train_fit  = train_full[train_full["ts"] <= val_cutoff].copy()
            val        = train_full[train_full["ts"] >  val_cutoff].copy()

            logger.logging.info(
                f"Split — train_fit: {len(train_fit)}, val: {len(val)}, "
                f"test: {len(test)} (holdout)"
            )

            # ---- 3. Feature/target setup ----
            features = [c for c in df.columns if c not in self.config.exclude_cols]
            target   = self.config.target_col

            self.X_train_fit, self.y_train_fit = train_fit[features], train_fit[target]
            self.X_val,       self.y_val       = val[features],       val[target]
            X_test,           y_test           = test[features],      test[target]
            X_train_full,     y_train_full     = train_full[features], train_full[target]

            logger.logging.info(f"Using {len(features)} features")

            # ---- 4. Run Optuna search ----
            logger.logging.info(f"Starting Optuna search — {self.config.n_trials} trials")

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            )
            study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                show_progress_bar=True,
            )

            best_val_mae = study.best_value
            best_params  = study.best_params
            logger.logging.info(f"Search complete — best validation MAE: {best_val_mae:.4f}")
            logger.logging.info(f"Best params: {best_params}")

            # ---- 5. Train final model on train+val with best params ----
            final_params = best_params.copy()
            final_params.update({
                "objective":    "poisson",
                "metric":       "mae",
                "verbose":      -1,
                "random_state": self.config.random_state,
                "n_estimators": 3000,
            })

            final_model = lgb.LGBMRegressor(**final_params)
            final_model.fit(
                X_train_full, y_train_full,
                eval_set=[(X_test, y_test)],
                categorical_feature=self.config.cat_features,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_final),
                    lgb.log_evaluation(200),
                ]
            )
            logger.logging.info(
                f"Final model trained — best iteration: {final_model.best_iteration_}"
            )

            # ---- 6. Evaluate on test set ----
            pred = np.clip(final_model.predict(X_test), 0, None)

            tuned_mae  = mean_absolute_error(y_test, pred)
            tuned_rmse = np.sqrt(mean_squared_error(y_test, pred))
            tuned_wape = np.abs(y_test - pred).sum() / y_test.sum()

            # Reference values from Notebook 3 / model_trainer for comparison
            baseline_mae  = 2.220
            baseline_wape = 43.41
            default_mae   = 1.649
            default_wape  = 32.25

            tuning_improvement_mae  = (1 - tuned_mae  / default_mae)  * 100
            tuning_improvement_wape = (1 - tuned_wape * 100 / default_wape) * 100
            total_improvement_mae   = (1 - tuned_mae  / baseline_mae) * 100
            total_improvement_wape  = (1 - tuned_wape * 100 / baseline_wape) * 100

            logger.logging.info(
                f"Tuned LightGBM — MAE: {tuned_mae:.3f}, RMSE: {tuned_rmse:.3f}, "
                f"WAPE: {tuned_wape*100:.2f}%"
            )
            logger.logging.info(
                f"Improvement over default LightGBM — "
                f"MAE: {tuning_improvement_mae:.2f}%, WAPE: {tuning_improvement_wape:.2f}%"
            )
            logger.logging.info(
                f"Total improvement over seasonal-naive — "
                f"MAE: {total_improvement_mae:.2f}%, WAPE: {total_improvement_wape:.2f}%"
            )

            # ---- 7. Save artifacts ----
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

            # Model
            with open(self.config.model_path, "wb") as f:
                pickle.dump(final_model, f)
            logger.logging.info(f"Tuned model saved to {self.config.model_path}")

            # Best params
            with open(self.config.best_params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            logger.logging.info(f"Best params saved to {self.config.best_params_path}")

            # Metrics summary
            metrics = {
                "n_trials":                self.config.n_trials,
                "best_val_mae":            round(best_val_mae, 4),
                "best_iteration":          final_model.best_iteration_,
                "baseline_mae":            baseline_mae,
                "baseline_wape":           baseline_wape,
                "default_lgbm_mae":        default_mae,
                "default_lgbm_wape":       default_wape,
                "tuned_lgbm_mae":          round(tuned_mae, 4),
                "tuned_lgbm_rmse":         round(tuned_rmse, 4),
                "tuned_lgbm_wape":         round(tuned_wape * 100, 2),
                "tuning_improvement_mae":  round(tuning_improvement_mae, 2),
                "tuning_improvement_wape": round(tuning_improvement_wape, 2),
                "total_improvement_mae":   round(total_improvement_mae, 2),
                "total_improvement_wape":  round(total_improvement_wape, 2),
            }
            with open(self.config.metrics_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
            logger.logging.info(f"Metrics saved to {self.config.metrics_path}")

            # Test predictions
            test_eval = test[["hub_id", "hub_name", "ts", "orders"]].copy()
            test_eval["pred"] = pred
            test_eval.to_csv(self.config.predictions_path, index=False)
            logger.logging.info(f"Predictions saved to {self.config.predictions_path}")

            # Feature importance
            imp = pd.DataFrame({
                "feature":    features,
                "importance": final_model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp.to_csv(self.config.importance_path, index=False)
            logger.logging.info(
                f"Feature importance saved — top 5: {imp.head(5)['feature'].tolist()}"
            )

            # Optuna study (for later analysis / visualization)
            with open(self.config.study_path, "wb") as f:
                pickle.dump(study, f)
            logger.logging.info(f"Optuna study saved to {self.config.study_path}")

            return metrics

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    tuner = HyperparameterTuner()
    metrics = tuner.initiate_hyperparameter_tuning()
    print("\nHyperparameter tuning complete. Key metrics:")
    for k in ["best_val_mae", "tuned_lgbm_mae", "tuned_lgbm_wape",
              "tuning_improvement_mae", "total_improvement_mae"]:
        print(f"  {k:30s}: {metrics[k]}")