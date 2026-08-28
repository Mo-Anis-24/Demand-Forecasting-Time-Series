import sys

from src.exception.exception import forcast
from src.logger import logger

from src.components.data_ingestion         import DataIngestion
from src.components.data_transformation    import DataTransformation
from src.components.model_trainer          import ModelTrainer
from src.components.hyperparameter_tuning  import HyperparameterTuner


class TrainPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logger.logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            ingestion_path = data_ingestion.initiate_data_ingestion()
            return ingestion_path
        except Exception as e:
            raise forcast(e, sys)

    def start_data_transformation(self):
        try:
            logger.logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            features_path = data_transformation.initiate_data_transformation()
            return features_path
        except Exception as e:
            raise forcast(e, sys)

    def start_model_training(self):
        try:
            logger.logging.info("Starting model training")
            model_trainer = ModelTrainer()
            metrics = model_trainer.initiate_model_trainer()
            return metrics
        except Exception as e:
            raise forcast(e, sys)

    def start_hyperparameter_tuning(self):
        try:
            logger.logging.info("Starting hyperparameter tuning")
            tuner = HyperparameterTuner()
            metrics = tuner.initiate_hyperparameter_tuning()
            return metrics
        except Exception as e:
            raise forcast(e, sys)

    def run_pipeline(self):
        try:
            logger.logging.info("=" * 60)
            logger.logging.info("TRAIN PIPELINE STARTED")
            logger.logging.info("=" * 60)

            ingestion_path   = self.start_data_ingestion()
            features_path    = self.start_data_transformation()
            trainer_metrics  = self.start_model_training()
            tuner_metrics    = self.start_hyperparameter_tuning()

            logger.logging.info("=" * 60)
            logger.logging.info("TRAIN PIPELINE COMPLETE")
            logger.logging.info("=" * 60)

            print("\n=== PIPELINE SUMMARY ===")
            print(f"Ingestion output : {ingestion_path}")
            print(f"Features output  : {features_path}")
            print(f"Default LightGBM : MAE {trainer_metrics['lgbm_mae']}, "
                  f"improvement {trainer_metrics['improvement_mae_pct']}%")
            print(f"Tuned LightGBM   : MAE {tuner_metrics['tuned_lgbm_mae']}, "
                  f"total improvement {tuner_metrics['total_improvement_mae']}%")

        except Exception as e:
            raise forcast(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()