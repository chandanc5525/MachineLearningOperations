# src/pipeline/pipeline.py
import yaml
from sklearn.model_selection import train_test_split
from src.data.data_ingestion import load_data
from src.features.feature_engineering import create_features
from src.models.model_train import train_model, save_model
from src.evaluation.evaluate_model import evaluate_model

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_pipeline():
    # Step 1: Load Data
    data = load_data()

    # Step 2: Feature Engineering
    df = create_features(data)

    # Step 3: Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['split']['test_size'],
        random_state=config['split']['random_state']
    )

    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    # Step 5: Train Model
    model = train_model(
        X_train, y_train,
        n_estimators=config['model']['n_estimators'],
        random_state=config['model']['random_state']
    )

    # Step 6: Evaluate Model
    r2, mse = evaluate_model(model, X_test, y_test)
    print(f"Model R2 Score: {r2:.4f}")
    print(f"Model MSE: {mse:.4f}")

    # Step 7: Save Model
    save_model(model, config['artifacts']['model_path'])

if __name__ == "__main__":
    run_pipeline()
