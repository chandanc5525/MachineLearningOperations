from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train Random Forest Regressor using Pipeline
    """
    numeric_features = X_train.columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    ])

    model.fit(X_train, y_train)
    return model

def save_model(model, filepath: str):
    """
    Save trained model using joblib
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved at: {filepath}")
