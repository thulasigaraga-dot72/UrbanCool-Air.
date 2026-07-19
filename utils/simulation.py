import os
import joblib
import pandas as pd
from utils.preprocess import preprocess_features
from utils.heat_score import compute_heat_risk

class DigitalTwinSimulator:
    def __init__(self, model_path="models/model.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                self.model = None
        else:
            print(f"Model path {self.model_path} does not exist.")
            self.model = None
            
    def run_simulation(self, baseline_data, modifications):
        """
        baseline_data: dict containing current state of the location
        modifications: dict containing variables adjusted by the user
        
        Returns:
            dict of baseline vs simulated results
        """
        # 1. Compute baseline LST (predicted by model, or actual if available. Let's predict both so comparison is apples-to-apples)
        baseline_df = preprocess_features(baseline_data)
        
        # 2. Apply modifications to create the simulated scenario
        simulated_data = baseline_data.copy()
        for key, val in modifications.items():
            if key in simulated_data:
                simulated_data[key] = val
                
        simulated_df = preprocess_features(simulated_data)
        
        if self.model is None:
            self._load_model()
            
        if self.model is not None:
            baseline_lst = float(self.model.predict(baseline_df)[0])
            simulated_lst = float(self.model.predict(simulated_df)[0])
        else:
            # Fallback to simple math if model is not trained yet
            print("Warning: ML model not loaded, running fallback simulation formula.")
            baseline_lst = float(baseline_data.get("lst", 30.0))
            # Rough approximation matching coefficients
            temp_diff = 0.0
            if "green_cover" in modifications:
                temp_diff -= 0.20 * (modifications["green_cover"] - baseline_data.get("green_cover", 0.0))
            if "water_body" in modifications:
                temp_diff -= 0.15 * (modifications["water_body"] - baseline_data.get("water_body", 0.0))
            if "albedo" in modifications:
                temp_diff -= 12.0 * (modifications["albedo"] - baseline_data.get("albedo", 0.0))
            if "building_density" in modifications:
                temp_diff += 0.18 * (modifications["building_density"] - baseline_data.get("building_density", 0.0))
            if "road_density" in modifications:
                temp_diff += 0.10 * (modifications["road_density"] - baseline_data.get("road_density", 0.0))
            simulated_lst = baseline_lst + temp_diff
            
        # 3. Compute Heat Risk Scores
        # For baseline
        base_risk = compute_heat_risk(
            lst=baseline_lst,
            building_density=float(baseline_data.get("building_density", 50.0)),
            aqi=float(baseline_data.get("aqi", 100.0)),
            humidity=float(baseline_data.get("humidity", 50.0)),
            green_cover=float(baseline_data.get("green_cover", 20.0)),
            wind_speed=float(baseline_data.get("wind_speed", 10.0))
        )
        
        # For simulated
        sim_risk = compute_heat_risk(
            lst=simulated_lst,
            building_density=float(simulated_data.get("building_density", 50.0)),
            aqi=float(simulated_data.get("aqi", 100.0)),
            humidity=float(simulated_data.get("humidity", 50.0)),
            green_cover=float(simulated_data.get("green_cover", 20.0)),
            wind_speed=float(simulated_data.get("wind_speed", 10.0))
        )
        
        return {
            "baseline_lst": round(baseline_lst, 2),
            "simulated_lst": round(simulated_lst, 2),
            "lst_change": round(simulated_lst - baseline_lst, 2),
            "baseline_risk_score": base_risk["score"],
            "baseline_risk_category": base_risk["category"],
            "baseline_risk_color": base_risk["color"],
            "baseline_risk_level": base_risk["level"],
            "simulated_risk_score": sim_risk["score"],
            "simulated_risk_category": sim_risk["category"],
            "simulated_risk_color": sim_risk["color"],
            "simulated_risk_level": sim_risk["level"],
            "risk_score_change": round(sim_risk["score"] - base_risk["score"], 2)
        }
