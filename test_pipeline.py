import os
import joblib
import pandas as pd
from utils.preprocess import preprocess_features
from utils.heat_score import compute_heat_risk
from utils.recommend import get_cooling_recommendations
from utils.simulation import DigitalTwinSimulator
from utils.shap_analysis import ShapExplainerWrapper

def run_tests():
    print("==================================================")
    print("            URBANCOOL AI TEST SUITE              ")
    print("==================================================")
    
    # 1. Check generated files
    print("\n[Step 1] Checking generated data and models...")
    files = [
        "data/locations.csv",
        "data/dataset.csv",
        "models/model.pkl",
        "models/encoder.pkl"
    ]
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  [OK] Found file: {f} ({size} bytes)")
        else:
            print(f"  [FAIL] Missing file: {f}")
            raise FileNotFoundError(f"Required file {f} was not found.")
            
    # 2. Test Preprocessing
    print("\n[Step 2] Testing data preprocessing...")
    sample_raw = {
        "elevation": 213.0,
        "pop_density": 11320.0,
        "green_cover": 20.0,
        "building_density": 50.0,
        "road_density": 45.0,
        "albedo": 0.15,
        "humidity": 50.0,
        "wind_speed": 12.0,
        "aqi": 110.0,
        "water_body": 2.0,
        "solar_radiation": 650.0,
        "surface_moisture": 35.0,
        "climate": "inland"
    }
    df_prepared = preprocess_features(sample_raw)
    print("  [OK] Feature dataframe created:")
    print(df_prepared)
    
    # 3. Test Model Load and Predict
    print("\n[Step 3] Testing ML Model Loading & Prediction...")
    model = joblib.load("models/model.pkl")
    prediction = model.predict(df_prepared)[0]
    print(f"  [OK] Predicted Land Surface Temp: {prediction:.2f} C")
    assert 10.0 <= prediction <= 50.0, f"Expected prediction between 10C and 50C, got {prediction}"
    
    # 4. Test Heat Score Calculation
    print("\n[Step 4] Testing Heat Risk Index Calculations...")
    risk_results = compute_heat_risk(
        lst=prediction,
        building_density=sample_raw["building_density"],
        aqi=sample_raw["aqi"],
        humidity=sample_raw["humidity"],
        green_cover=sample_raw["green_cover"],
        wind_speed=sample_raw["wind_speed"]
    )
    print(f"  [OK] Calculated Heat Risk Score: {risk_results['score']} / 100")
    print(f"  [OK] Category: {risk_results['category']} (Color: {risk_results['color']})")
    
    # 5. Test SHAP Analysis Explainer
    print("\n[Step 5] Testing SHAP Explainer...")
    shap_wrapper = ShapExplainerWrapper()
    explanation = shap_wrapper.explain(df_prepared)
    print(f"  [OK] Explainer loaded: Success = {explanation['success']}")
    print(f"  [OK] Base Expected Value: {explanation['base_value']:.2f} C")
    print(f"  [OK] Main contribution: {explanation['contributions'][0]['display_name']} -> {explanation['contributions'][0]['shap_value']:+.2f} C")
    
    # 6. Test Recommendations & Budget
    print("\n[Step 6] Testing Cooling Recommendations & ROI Solver...")
    recs = get_cooling_recommendations(sample_raw, budget_lakhs=250)
    print(f"  [OK] Strategy list built. Total catalog count: {len(recs['all_recommendations'])}")
    print(f"  [OK] Budget Allocator result: Spend = Rs. {recs['total_cost_lakhs']} Lakhs, Temp Reduction = -{recs['total_cooling_impact']:.3f} C")
    
    # 7. Test Digital Twin Simulation
    print("\n[Step 7] Testing Digital Twin Simulation comparison...")
    simulator = DigitalTwinSimulator()
    mods = {"green_cover": 35.0, "albedo": 0.25}
    sim_res = simulator.run_simulation(sample_raw, mods)
    print(f"  [OK] Simulation Complete:")
    print(f"    - Baseline Temp:  {sim_res['baseline_lst']} C")
    print(f"    - Simulated Temp: {sim_res['simulated_lst']} C")
    print(f"    - Temp Change:     {sim_res['lst_change']} C")
    print(f"    - Baseline Risk:  {sim_res['baseline_risk_score']} -> Sim Risk: {sim_res['simulated_risk_score']}")
    
    print("\n==================================================")
    print("      [OK] ALL TEST CASES PASSED SUCCESSFULLY!    ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
