import os
import joblib
import pandas as pd

DEFAULT_CLIMATE_MAP = {
    'inland': 0,
    'coastal': 1,
    'plateau': 2,
    'desert': 3,
    'mountain': 4,
    'coastal_plain': 5
}

def load_encoder():
    encoder_path = "models/encoder.pkl"
    if os.path.exists(encoder_path):
        try:
            return joblib.load(encoder_path)
        except Exception:
            return DEFAULT_CLIMATE_MAP
    return DEFAULT_CLIMATE_MAP

def preprocess_features(data_dict):
    """
    Converts a dictionary of features into a pandas DataFrame ready for ML prediction.
    Features order must match:
    [
      "elevation", "pop_density", "green_cover", "building_density",
      "road_density", "albedo", "humidity", "wind_speed", "aqi",
      "water_body", "solar_radiation", "surface_moisture", "climate_encoded"
    ]
    """
    climate_map = load_encoder()
    
    # Get numeric climate
    climate_str = data_dict.get("climate", "inland")
    climate_encoded = climate_map.get(climate_str.lower(), 0)
    
    prepared_dict = {
        "elevation": [float(data_dict.get("elevation", 0.0))],
        "pop_density": [float(data_dict.get("pop_density", 0.0))],
        "green_cover": [float(data_dict.get("green_cover", 0.0))],
        "building_density": [float(data_dict.get("building_density", 0.0))],
        "road_density": [float(data_dict.get("road_density", 0.0))],
        "albedo": [float(data_dict.get("albedo", 0.15))],
        "humidity": [float(data_dict.get("humidity", 50.0))],
        "wind_speed": [float(data_dict.get("wind_speed", 10.0))],
        "aqi": [float(data_dict.get("aqi", 100.0))],
        "water_body": [float(data_dict.get("water_body", 0.0))],
        "solar_radiation": [float(data_dict.get("solar_radiation", 500.0))],
        "surface_moisture": [float(data_dict.get("surface_moisture", 30.0))],
        "climate_encoded": [climate_encoded]
    }
    
    # Explicit column ordering
    columns = [
        "elevation", "pop_density", "green_cover", "building_density",
        "road_density", "albedo", "humidity", "wind_speed", "aqi",
        "water_body", "solar_radiation", "surface_moisture", "climate_encoded"
    ]
    
    return pd.DataFrame(prepared_dict)[columns]
