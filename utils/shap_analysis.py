import os
import joblib
import pandas as pd
import numpy as np

class ShapExplainerWrapper:
    def __init__(self, model_path="models/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self._initialize()
        
    def _initialize(self):
        if not os.path.exists(self.model_path):
            print("Model path does not exist. Cannot initialize SHAP explainer.")
            return
            
        try:
            self.model = joblib.load(self.model_path)
            
            # Import SHAP dynamically
            import shap
            
            # TreeExplainer is highly optimized for Random Forest Regressors
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
            
    def explain(self, features_df):
        """
        Calculates feature contributions for a specific scenario.
        Returns:
            dict containing base value, final prediction, list of contributions, and success flag.
        """
        display_names = {
            "elevation": "Elevation (m)",
            "pop_density": "Pop. Density (people/km²)",
            "green_cover": "Green Cover (%)",
            "building_density": "Building Density (%)",
            "road_density": "Road Density (%)",
            "albedo": "Albedo Score",
            "humidity": "Humidity (%)",
            "wind_speed": "Wind Speed (km/h)",
            "aqi": "Air Quality Index (AQI)",
            "water_body": "Water Body Cover (%)",
            "solar_radiation": "Solar Radiation (W/m²)",
            "surface_moisture": "Surface Moisture (%)",
            "climate_encoded": "Climate Zone Encoding"
        }
        
        if self.model is None or self.explainer is None:
            self._initialize()
            
        if self.model is not None and self.explainer is not None:
            try:
                # Compute SHAP values
                shap_values = self.explainer.shap_values(features_df)
                
                # Resolve shapes for different SHAP API versions
                if isinstance(shap_values, list):
                    s_vals = shap_values[0][0]
                elif len(shap_values.shape) == 3:  # (n_samples, n_features, n_outputs)
                    s_vals = shap_values[0, :, 0]
                else:
                    s_vals = shap_values[0]  # (n_samples, n_features)
                    
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
                    
                prediction = self.model.predict(features_df)[0]
                
                contributions = []
                for col, val, s_val in zip(features_df.columns, features_df.values[0], s_vals):
                    contributions.append({
                        "feature": col,
                        "display_name": display_names.get(col, col),
                        "actual_value": float(val),
                        "shap_value": float(s_val)
                    })
                    
                # Sort contributions by absolute impact
                contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
                
                return {
                    "base_value": float(base_value),
                    "prediction": float(prediction),
                    "contributions": contributions,
                    "success": True
                }
            except Exception as e:
                print(f"Error computing SHAP values: {e}. Falling back to heuristics.")
                
        # Heuristic/Fallback model explanation based on physical weights from data_generator.py
        # Used if SHAP fails or model is not loaded yet
        base_val = 24.5
        prediction = 24.5
        
        means = {
            "elevation": 500.0, "pop_density": 5000.0, "green_cover": 20.0, "building_density": 50.0,
            "road_density": 40.0, "albedo": 0.18, "humidity": 55.0, "wind_speed": 10.0, "aqi": 100.0,
            "water_body": 5.0, "solar_radiation": 600.0, "surface_moisture": 30.0, "climate_encoded": 1.0
        }
        
        coeffs = {
            "elevation": -0.003, "pop_density": 0.00001, "green_cover": -0.20, "building_density": 0.18,
            "road_density": 0.10, "albedo": -12.0, "humidity": -0.05, "wind_speed": -0.15, "aqi": 0.02,
            "water_body": -0.15, "solar_radiation": 0.015, "surface_moisture": -0.10, "climate_encoded": 0.0
        }
        
        contributions = []
        for col in features_df.columns:
            val = float(features_df[col].values[0])
            mean_val = means.get(col, 0.0)
            coeff = coeffs.get(col, 0.0)
            s_val = (val - mean_val) * coeff
            
            contributions.append({
                "feature": col,
                "display_name": display_names.get(col, col),
                "actual_value": val,
                "shap_value": s_val
            })
            prediction += s_val
            
        contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "base_value": base_val,
            "prediction": prediction,
            "contributions": contributions,
            "success": False
        }
