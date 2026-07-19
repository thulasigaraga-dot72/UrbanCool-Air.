import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CLIMATE_MAP = {
    'inland': 0,
    'coastal': 1,
    'plateau': 2,
    'desert': 3,
    'mountain': 4,
    'coastal_plain': 5
}

def train():
    print("Loading dataset...")
    if not os.path.exists("data/dataset.csv"):
        raise FileNotFoundError("Dataset not found. Please run data_generator.py first.")
        
    df = pd.read_csv("data/dataset.csv")
    
    # Map climate strings to numeric
    df['climate_encoded'] = df['climate'].map(CLIMATE_MAP)
    
    # Features & Target
    features = [
        "elevation", "pop_density", "green_cover", "building_density",
        "road_density", "albedo", "humidity", "wind_speed", "aqi",
        "water_body", "solar_radiation", "surface_moisture", "climate_encoded"
    ]
    target = "lst"
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest Regressor on {X_train.shape[0]} samples...")
    model = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared Score (R2):     {r2:.4f}")
    print("------------------------\n")
    
    # Save model and encoder/mappings
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(CLIMATE_MAP, "models/encoder.pkl")
    print("Model saved to models/model.pkl")
    print("Encoder map saved to models/encoder.pkl")

if __name__ == "__main__":
    train()
