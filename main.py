from src.pipeline.pipeline import run_pipeline
from src.data.data_ingestion import load_data
from src.features.feature_engineering import create_features
from src.models.model_train import train_model, save_model
from src.evaluation.evaluate_model import evaluate_model

if __name__ == "__main__":
    run_pipeline()
