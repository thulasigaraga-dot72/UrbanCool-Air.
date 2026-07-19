# Project Report: UrbanCool AI
*Smart City Decision Support Platform for Urban Heat Island (UHI) Mitigation*

UrbanCool AI is an interactive, data-driven planning platform designed to analyze, explain, simulate, and recommend urban cooling strategies. It enables city planners to test infrastructure changes in a digital twin environment, evaluate localized heat risk indexes, calculate the ROI of mitigation strategies, and get real-time explainability insights.

---

## 1. System Architecture

UrbanCool AI is structured as a modular Python application composed of a data generation layer, a machine learning model pipeline, a backend engine suite, and an interactive Streamlit UI dashboard.

```mermaid
graph TD
    subgraph Data Layer
        A[data_generator.py] -->|Generates| B[locations.csv]
        A -->|Generates| C[dataset.csv]
    end
    subgraph Model Layer
        C -->|Train| D[train_model.py]
        D -->|Saves| E[model.pkl]
        D -->|Saves| F[encoder.pkl]
    end
    subgraph Core Analytical Engines
        E & F --> Pre[preprocess.py]
        E --> Sim[simulation.py]
        E --> Shap[shap_analysis.py]
        Heat[heat_score.py]
        Rec[recommend.py]
    end
    subgraph UI Layer (Streamlit App)
        App[app.py] -->|Incorporate| Pre & Sim & Shap & Heat & Rec
        App -->|Renders| UI[User Interface]
        UI -->|Interactions| Map[India Heat Map]
        UI -->|What-If| Twin[Digital Twin Console]
        UI -->|Explain| SHAP_UI[SHAP Bar Chart]
        UI -->|Optimize| Budget[Budget ROI Solver]
        UI -->|Consult| Chat[AI Planner Assistant]
    end
```

### Component Details
1. **Data Generator (`utils/data_generator.py`)**: Models 50 major Indian cities (with geographic coordinate baselines, elevations, and population densities) and creates a synthetic dataset of 20,000 environmental scenarios.
2. **ML Pipeline (`models/train_model.py`)**: Encodes climate zones, splits data into training/testing sets, trains a regressor to predict Land Surface Temperature (LST), and dumps files to disk.
3. **Core Utility Suite (`utils/`)**:
   - `preprocess.py`: Handles raw inputs, mapping climate zones to numeric indices and formatting feature vectors.
   - `heat_score.py`: Normalizes factors and calculates a unified Heat Risk Index (0–100).
   - `recommend.py`: Priority-ranks cooling strategies, matching them to a defined budget via a greedy knapsack optimizer.
   - `simulation.py`: Serves as the "What-If" digital twin engine, evaluating modifications against baselines.
   - `shap_analysis.py`: Dynamically extracts Shapley values to trace the model's inner decision-making.
4. **Front-End Portal (`dashboard/app.py`)**: An authenticated split-screen dashboard displaying maps, gauges, simulation consoles, explainability charts, and chat interfaces.

---

## 2. Technology Stack & Model Decisions

UrbanCool AI is built with a pure Python data science stack designed for speed, flexibility, and easy local deployment.

| Tech Stack Component | Selected Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Standard language for data processing, machine learning, and visualization libraries. |
| **User Interface** | Streamlit | Rapid development of premium, state-of-the-art interactive dashboards. |
| **Core ML Model** | `RandomForestRegressor` (Scikit-Learn) | Predicts Land Surface Temperature (LST) based on physical and environmental inputs. |
| **Model Explainability** | SHAP (SHapley Additive exPlanations) | Computes feature contribution values using `shap.TreeExplainer` for model transparency. |
| **Interactive Mapping** | Folium / Streamlit-Folium | Renders an interactive map of India with colored regional markers. |
| **Data Visualizations** | Plotly (Gauge, Scatter, Bar, Radar) | Provides interactive, animated, high-fidelity metrics visualizations. |
| **Serialization** | Joblib | Exports and loads trained models (`model.pkl`) and categorical encoders (`encoder.pkl`). |

### Machine Learning Model: Random Forest Regressor
The system uses a **Random Forest Regressor** trained on 20,000 synthetic urban microclimate variations.

#### Why Random Forest?
* **Non-Linear Relationships**: The thermodynamic equations governing land surface heating (solar radiation, evaporative cooling from tree canopies, heat retention by high-albedo/low-albedo surfaces) are highly non-linear. Random Forest excels at building split boundaries that capture non-linear features without requiring explicit kernel mappings.
* **Collinearity Handling**: Urban indicators (such as population density, road density, and building density) are naturally highly correlated. Traditional linear models suffer from multicollinearity issues; Random Forest mitigates this by randomly sampling subsets of features at each node split.
* **High Efficiency & Speed**: With a tuned configuration (`n_estimators=80`, `max_depth=12`), predictions execute in milliseconds. This is critical for supporting real-time "What-If" sliders in the UI.
* **Tree SHAP Compatibility**: Unlike deep learning models which require slow perturbation-based calculations (Kernel SHAP), tree ensembles are fully compatible with **Tree SHAP** (`shap.TreeExplainer`). This algorithm runs in \(O(TLD^2)\) time, enabling the dashboard to render exact feature contribution breakdowns instantly.

---

## 3. Key Features

* **Authenticated Access System**: A premium split-screen portal with branding and Google OAuth mock button, securing access to municipal planning tools.
* **Interactive Geographic Heat Map**: Displays 50 Indian planning locations colored by LST/Heat Risk. Planners click pins or use the selector to load baseline city data.
* **Digital Twin Sandbox Console**: Sliders allow planners to modify urban parameters (Green Cover, Building Density, Albedo, Water Cover) and immediately see "simulated" future LST and Heat Risk changes.
* **Explainable AI (SHAP Waterfall)**: Renders a horizontal bar chart showing how much each factor (e.g., canopy loss or pavement density) pushed the local temperature above or below the baseline mean.
* **Cooling Recommendations & ROI Solver**: Computes costs, projected temperature drops, and ROI (Cooling per Lakh Rupees) for 5 core interventions. It includes an optimizer where planners enter a budget limit to find the most cost-efficient mitigation portfolio.
* **Contextual AI Urban Planner Assistant**: An interactive chat interface that parses queries and provides tailored climate diagnosis based on the city's climate zone (e.g., desert advice for Jaipur, coastal advice for Mumbai).

---

## 4. Advantages & Benefits

### Technical & Operational Advantages
* **Explainable AI (XAI)**: Eliminates "black-box" model distrust. Planners see exactly *why* the model makes a prediction, boosting confidence in administrative decision-making.
* **Sub-second Simulations**: Pre-cached tree explainers and model instances enable instant recalculations on parameter adjustments.
* **Modular Integration**: Utility functions (`preprocess.py`, `recommend.py`) are fully separated from the UI, meaning they can easily be exposed via a REST API or scheduled background jobs.
* **Low Computational Overhead**: Runs entirely on a standard CPU container or local workstation without requiring expensive GPU infrastructure.

### Societal & Environmental Benefits
* **Optimized Resource Allocation**: Helps city governments avoid waste by directing budgets to high-ROI actions (e.g., showing that cool roofs have higher cooling impact per rupee in dense areas than planting trees).
* **Mitigating Urban Heat Islands (UHI)**: Aids in bringing down microclimate temperatures, directly reducing heat stroke incidents and related public health risks.
* **Lower Peak Energy Demand**: Lower ambient temperatures reduce cooling loads (AC usage) in surrounding buildings, leading to reduced carbon emissions.
* **UN SDG 11 Alignment**: Directly supports the United Nations Sustainable Development Goal 11 to make cities inclusive, safe, resilient, and sustainable.

---

## 5. Limitations

* **Synthetic Data Base**: The models are currently trained on synthetic datasets generated via physical approximations. While these capture key thermodynamic dynamics (e.g., albedo vs. LST), they must be calibrated with real satellite (Landsat/MODIS) and weather station data for actual municipal deployment.
* **Fixed Spatial Resolution**: The recommendations and costs are calculated for a standardized 1 sq km urban sector. It does not account for detailed micro-level fluid dynamics, wind-tunnel effects between skyscrapers, or topographical valleys.
* **Rule-Based Conversational Chat**: The AI assistant uses dynamic keyword routing and pattern matching. It does not have the open-ended reasoning capabilities of a Large Language Model (LLM) API, although it can be easily upgraded to one.

---

## 6. Target End Users

1. **Urban Planners & Municipal Corporations**: Officials in municipal bodies (e.g., DDA, BMC, BBMP) seeking to evaluate smart-city zoning laws, forestry mandates, and rooftop regulations.
2. **Environmental & Climate Consultants**: Advisors conducting environmental impact assessments (EIAs) for new special economic zones (SEZs) or residential layouts.
3. **Civil Architects & Sustainability Officers**: Designers looking to calculate the cooling and environmental ROI of high-albedo paints, green roofs, and porous pavements for large commercial buildings.
4. **Smart City Project Developers**: Teams managing budget allocations for public spaces, parks, and urban renewal projects.
5. **Research & Academic Bodies**: Environmental scientists studying climate resilience strategies in South Asian cities.
