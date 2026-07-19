# UrbanCool AI - Implementation Plan

UrbanCool AI is a Smart City Decision Support Platform designed to analyze, explain, simulate, and recommend urban heat island (UHI) mitigation strategies. This implementation plan outlines the steps to build the synthetic datasets, train the Machine Learning model, implement custom analytical engines, and assemble a premium, interactive Streamlit dashboard.

---

## Technical Architecture Overview

The system consists of the following components:
1. **Synthetic Data Generator**: Generates 50 Indian locations with realistic coordinates (latitude/longitude), elevation, and baseline urban features. It then builds a dataset of 20,000 environmental scenarios (50 locations × 400 scenarios or similar split to reach ~20,000 samples) mapping land surface temperature (LST) to inputs based on physical formulas.
2. **ML Modeling Pipeline**:
   - Random Forest Regressor trained on the synthetic dataset.
   - Saves `model.pkl` and any preprocessing pipelines (using `joblib`).
3. **Core Analytical Engines (in `utils/`)**:
   - `preprocess.py`: Data transformations and preprocessing.
   - `heat_score.py`: Computes the Heat Risk Index using a multi-factor formula.
   - `recommend.py`: Evaluates weighted cooling strategies, costs, and ROI.
   - `simulation.py`: Computes "what-if" scenarios for digital twin planning.
   - `shap_analysis.py`: Performs SHAP calculations for explanation.
4. **Streamlit Interactive Dashboard**:
   - Modern dark mode styling with premium custom CSS.
   - Visualizations using Plotly (Gauge, Radar, Bar, Scatter, Line).
   - India Heat Map using Folium.
   - AI Urban Planner Assistant (rule-based diagnostic agent).

---

## Proposed Changes

We will create the project structure in `c:\gitam`.

### 1. Requirements and Configurations

#### [NEW] [requirements.txt](file:///c:/gitam/requirements.txt)
Specifies dependencies for data generation, machine learning, explainability, mapping, and UI:
- `numpy`, `pandas`
- `scikit-learn`
- `shap`
- `streamlit`, `streamlit-folium`
- `folium`
- `plotly`
- `joblib`

---

### 2. Data and Model Generation

#### [NEW] [data_generator.py](file:///c:/gitam/utils/data_generator.py)
Generates:
- `data/locations.csv`: 50 diverse cities/neighborhoods across India (representing various states and climate zones: e.g., Delhi, Mumbai, Kolkata, Chennai, Bengaluru, Hyderabad, Visakhapatnam, Jaipur, Srinagar, Ahmedabad, Guwahati, Pune, etc.) with their respective coordinates, baseline elevation, population density, and baseline green cover.
- `data/dataset.csv`: 20,000 samples. For each of the 50 locations, we generate 400 scenarios by varying features like:
  - Green Cover (0% to 60%)
  - Building Density (10% to 90%)
  - Road Density (10% to 90%)
  - Albedo (0.1 to 0.45)
  - Humidity (30% to 90%)
  - Wind Speed (2 km/h to 25 km/h)
  - AQI (30 to 350)
  - Water Body (0% to 20%)
  - Solar Radiation (200 to 1000 W/m²)
  - Surface Moisture (5% to 80%)
- The target **Land Surface Temperature (LST)** will be calculated using a multi-variable physical-approximation formula:
  $$\text{LST} = 22.0 + 0.015 \times \text{SolarRadiation} + 0.18 \times \text{BuildingDensity} + 0.10 \times \text{RoadDensity} + 0.01 \times \text{PopDensity}/1000 - 0.20 \times \text{GreenCover} - 0.15 \times \text{WaterBody} - 12.0 \times \text{Albedo} - 0.15 \times \text{WindSpeed} - 0.05 \times \text{Humidity} - 0.003 \times \text{Elevation} + 0.02 \times \text{AQI} - 0.10 \times \text{SurfaceMoisture} + \epsilon$$
  where $\epsilon \sim \mathcal{N}(0, 0.5)$ adds realistic noise. We will also adjust base temperatures according to each city's regional climate (e.g. desert regions like Jaipur will have higher baseline LST, while montane regions like Srinagar will have lower baseline LST).

#### [NEW] [train_model.py](file:///c:/gitam/models/train_model.py)
Trains a Scikit-Learn `Random Forest Regressor` on the generated dataset:
- Splits data (80% train, 20% test).
- Tracks regression metrics ($R^2$, MSE, MAE).
- Fits model and exports `models/model.pkl`.
- Fits and exports feature encoders (if categorical features are used) to `models/encoder.pkl`.

---

### 3. Utility Modules

#### [NEW] [preprocess.py](file:///c:/gitam/utils/preprocess.py)
Utility functions to clean, scale, and format input features for both modeling and simulation.

#### [NEW] [heat_score.py](file:///c:/gitam/utils/heat_score.py)
Computes the **Heat Risk Index** on a 0-100 scale:
- Formula:
  $$\text{Heat Risk} = w_1 \times \text{LST} + w_2 \times \text{BuildingDensity} + w_3 \times \text{AQI} + w_4 \times \text{Humidity} - w_5 \times \text{GreenCover} - w_6 \times \text{WindSpeed}$$
  (appropriately normalized and clamped between 0 and 100).
- Categorizes risk:
  - **< 35**: Cool / Low Risk (Green)
  - **35 - 55**: Moderate Risk (Yellow)
  - **55 - 75**: High Risk (Orange)
  - **> 75**: Extreme Risk (Red)

#### [NEW] [recommend.py](file:///c:/gitam/utils/recommend.py)
Implements the rule-based weighted cooling recommendations engine:
- Evaluates constraints (e.g. if Green Cover is low, recommend Planting Trees; if Building Density is high, recommend Green Roofs).
- Computes estimated cooling impact and implementation cost based on standard unit rates (e.g., ₹1000 per tree, ₹1200 per sq. meter of cool roof, etc.) adjusted for the population size/area density of the selected location.
- Outputs prioritized interventions sorted by ROI (Cooling per Lakh Rupees).

#### [NEW] [simulation.py](file:///c:/gitam/utils/simulation.py)
Handles digital twin simulations:
- Compares user-modified inputs to baseline inputs.
- Formulates a feature vector for the future scenario, calls the Random Forest model to predict the new LST, and computes differences in LST and Heat Risk Index.

#### [NEW] [shap_analysis.py](file:///c:/gitam/utils/shap_analysis.py)
Uses the `shap` library to calculate SHAP explanation values:
- Computes SHAP values on a single test prediction dynamically when a user clicks a location.
- Formats SHAP contributions as plus/minus temperature shifts for clear user messaging.

---

### 4. Interactive Dashboard

#### [NEW] [app.py](file:///c:/gitam/dashboard/app.py)
The core frontend application structured as follows:
- **Custom Theme (CSS)**: Sleek dark-mode, neon accents, curved cards, shadow hover effects, glassmorphic inputs.
- **Top Row**: Title, logo, SDG badges, and global metrics (Hottest locations ranking).
- **Interactive Map**: A Folium map of India displaying the 50 locations colored by their current LST / Heat Risk. Clicking a marker or choosing from a sidebar selectbox dynamically updates the active planning location.
- **Sidebar (Planning Panel)**: Features for adjusting environmental variables (Green Cover, Building Density, Albedo, AQI, Water Body, etc.) to perform **Digital Twin Simulations**.
- **Main View (Dashboard Tabs)**:
- **Tab 1: Overview & Prediction**:
  - Dual metric cards showing Before vs. After temperature and Heat Risk.
  - Custom Plotly Gauge charts for Heat Risk Index.
  - Future 7-day temperature trends (simulated weather forecasting).
- **Tab 2: AI Planner & Explainer**:
  - SHAP waterfall plot showing exactly why the model predicted that temperature (Explainable AI).
  - Radar Chart showing Current vs. Ideal Urban profile.
  - **AI Urban Planner Assistant**: Conversational output showing a tailored diagnosis ("Delhi is hot because...") and recommending actionable items.
- **Tab 3: cooling Recommendations & ROI**:
  - Prioritized table/list of recommended interventions with estimated costs, cooling impact, and ROI.
  - Plotly Scatter chart (Cost vs. Cooling Impact) and Bar chart (Cooling Strategy Impact).
  - Budget Planner: User input for maximum budget limits to auto-select the best combination of strategies.

---

## Verification Plan

### Automated Verification
We will run:
1. `python utils/data_generator.py` to verify data generation.
2. `python models/train_model.py` to train and save the model, verifying $R^2 > 0.90$.
3. A test script `test_pipeline.py` to verify all utilities (preprocessing, SHAP, cooling recommendations, heat risk index).

### Manual Verification
1. Launch the Streamlit dashboard using:
   `streamlit run dashboard/app.py`
2. Open the dashboard in the browser and test:
   - Clicking map pins and updating variables.
   - Running Digital Twin simulations and verifying that modifying sliders (e.g. increasing green cover) drops the predicted temperature.
   - Adjusting budget and verifying recommended cooling options dynamically update.
   - Checking SHAP waterfall plots and trend line graphs.

---

## Open Questions & Review Required

> [!NOTE]
> 1. **SHAP Speed Optimization**: SHAP tree-explainers can take a couple of seconds to initialize. To keep the app snappy, we will pre-initialize the SHAP explainer on the trained Random Forest model and cache it using Streamlit's `@st.cache_resource`.
> 2. **Map Click Events**: To make map click events seamless, we will use a sidebar selectbox synchronised with `streamlit-folium` marker click data. If `streamlit-folium` click events are slow, the selectbox serves as a perfect instant fallback.
