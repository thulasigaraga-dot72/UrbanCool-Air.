import os
import sys

# Add project root to python path to resolve utils imports correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

# Import core analytical modules
from utils.preprocess import preprocess_features
from utils.heat_score import compute_heat_risk
from utils.recommend import get_cooling_recommendations
from utils.simulation import DigitalTwinSimulator
from utils.shap_analysis import ShapExplainerWrapper

# Page Configuration
st.set_page_config(
    page_title="UrbanCool AI - Professional Planning Portal",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_city" not in st.session_state:
    st.session_state.selected_city = "Delhi"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome! I am the AI Urban Planner assistant. Select any location or adjust parameters to start your heat diagnostic query. How can I help you cool your city today?"}
    ]
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

# Dynamic Color Palette variables based on Theme selection
if st.session_state.light_mode:
    bg_color = "#f8fafc"
    text_color = "#0f172a"
    card_bg = "#ffffff"
    card_border = "rgba(0, 0, 0, 0.08)"
    card_shadow = "0 4px 15px rgba(0,0,0,0.06)"
    hover_border = "rgba(16, 185, 129, 0.4)"
    title_color = "#0f172a"
    sec_text_color = "#475569"
    sidebar_bg = "#f1f5f9"
    sidebar_text = "#0f172a"
    metric_text = "#0f172a"
    sub_bg = "#f1f5f9"
else:
    bg_color = "#0b0d10"
    text_color = "#e2e8f0"
    card_bg = "rgba(255, 255, 255, 0.02)"
    card_border = "rgba(255, 255, 255, 0.05)"
    card_shadow = "0 4px 20px rgba(0, 0, 0, 0.4)"
    hover_border = "rgba(16, 185, 129, 0.2)"
    title_color = "#ffffff"
    sec_text_color = "#94a3b8"
    sidebar_bg = "#0c0f13"
    sidebar_text = "#e2e8f0"
    metric_text = "#ffffff"
    sub_bg = "rgba(0,0,0,0.2)"

# Custom Premium Styling with Larger Fonts
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Main body background & font */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Global Font Size Enhancements */
    p, span, label, li, td, th {{
        font-size: 15px !important;
        line-height: 1.6 !important;
    }}
    
    /* Headers styling */
    h1 {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 38px !important;
        color: {title_color} !important;
        letter-spacing: -0.02em;
    }}
    h2 {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 30px !important;
        color: {title_color} !important;
    }}
    h3 {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 22px !important;
        color: {title_color} !important;
    }}
    h4 {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 18px !important;
        color: {title_color} !important;
    }}
    
    /* Container block overrides */
    div[data-testid="stVerticalBlockBorderOnly"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: {card_shadow} !important;
        backdrop-filter: blur(10px) !important;
        margin-bottom: 20px !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }}
    
    div[data-testid="stVerticalBlockBorderOnly"]:hover {{
        transform: translateY(-2px);
        border-color: {hover_border} !important;
    }}
    
    /* Sidebar premium branding footer */
    .brand-footer {{
        font-size: 12px;
        color: {sec_text_color};
        text-align: center;
        margin-top: 40px;
        border-top: 1px solid {card_border};
        padding-top: 15px;
    }}
    
    /* Brand indicator badge */
    .ai-badge {{
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }}
    
    /* Landmark image styles */
    .landmark-card {{
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 15px;
        border: 1px solid {card_border};
    }}
    
    /* Custom split-screen login branding container */
    .login-brand-container {{
        background: linear-gradient(rgba(11, 13, 16, 0.85), rgba(11, 13, 16, 0.95)), 
                    url("https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=1200&q=80");
        background-size: cover;
        background-position: center;
        padding: 80px 40px;
        border-radius: 24px;
        min-height: 600px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    
    /* Sidebar selectors */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {card_border} !important;
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: {sidebar_text} !important;
        font-weight: 500;
    }}
    
    /* Input formatting override */
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        color: {text_color} !important;
        font-size: 15px !important;
    }}
</style>
""", unsafe_allow_html=True)

# Helper: Load locations
@st.cache_data
def load_locations():
    if os.path.exists("data/locations.csv"):
        return pd.read_csv("data/locations.csv")
    return pd.DataFrame()

df_cities = load_locations()
if df_cities.empty:
    st.error("Error: Locations database not found. Please run data generator.")
    st.stop()

# Helper: Model pipelines
@st.cache_resource
def get_simulator():
    return DigitalTwinSimulator()

@st.cache_resource
def get_shap_wrapper():
    return ShapExplainerWrapper()

# ================= LOGIN INTERFACE (SPLIT SCREEN) =================
if not st.session_state.logged_in:
    # Hide default sidebar & header elements in login view
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stHeader"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    col_brand, col_login = st.columns([1.2, 1])
    
    with col_brand:
        st.markdown(f"""
        <div class="login-brand-container">
            <h1 style="color: #1E3A8A; font-size: 56px; font-weight: 800; margin-bottom: 12px; letter-spacing: -0.03em;">UrbanCool AI</h1>
            <p style="color: #94a3b8; font-size: 19px; max-width: 480px; line-height: 1.6; margin: 0 auto 30px auto;">
                Smart City Heat Mitigation Portal & Digital Twin Simulator. 
                Combatting Urban Heat Islands using Explainable AI models.
            </p>
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); padding: 20px; border-radius: 12px; max-width: 400px; text-align: left; backdrop-filter: blur(8px);">
                <h5 style="color: #ffffff; margin-top: 0; margin-bottom: 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #1E3A8A;">🔋 Active Analytics Platform</h5>
                <p style="color: #94a3b8; font-size: 14px; margin-bottom: 6px;">🟢 Random Forest Temperature Regressor</p>
                <p style="color: #94a3b8; font-size: 14px; margin-bottom: 6px;">🟢 Multi-Factor Evapotranspiration Heat Score</p>
                <p style="color: #94a3b8; font-size: 14px; margin-bottom: 0;">🟢 Dynamic SHAP Local Explainers</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_login:
        st.markdown("<div style='padding: 40px 10px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin-bottom: 0; font-size: 34px; font-weight: 800;'>Intelligence Console</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 15px; margin-top: 5px; margin-bottom: 30px;'>Enter your administrative credentials to log in.</p>", unsafe_allow_html=True)
        
        login_email = st.text_input("Email Address", value="admin@urbancool.ai")
        login_password = st.text_input("Password", type="password", value="admin123")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_rem, col_forg = st.columns(2)
        with col_rem:
            st.checkbox("Keep me signed in", value=True)
        with col_forg:
            st.markdown("<p style='text-align: right; color: #818cf8; font-size: 14px; margin-top: 3px; cursor: pointer;'>Forgot password?</p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        login_submit = st.button("Sign In to Portal", type="primary", use_container_width=True)
        google_submit = st.button("🌐 Continue with Google Workspace", use_container_width=True)
        
        st.markdown(
            "<div style='border: 1px dashed rgba(255,255,255,0.08); padding: 15px; border-radius: 8px; margin-top: 30px; background: rgba(255,255,255,0.01);'>"
            "<p style='color: #10B981; font-size: 13px; margin: 0; font-weight: 600;'>💡 Demonstration Credentials:</p>"
            "<p style='color: #94a3b8; font-size: 13px; margin: 2px 0 0 0;'>Email: <code>admin@urbancool.ai</code> | Password: <code>admin123</code></p>"
            "<p style='color: #64748b; font-size: 12px; margin: 5px 0 0 0;'>Any input will bypass to facilitate exploration.</p>"
            "</div>",
            unsafe_allow_html=True
        )
        
        if login_submit or google_submit:
            if login_email and login_password:
                st.session_state.logged_in = True
                st.success("Access granted!")
                st.rerun()
            else:
                st.error("Please fill in both email and password fields.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ================= DASHBOARD INTERFACE (LOGGED IN) =================

# Active City selector synchronised with Session state
city_options = df_cities["city"].tolist()
active_index = city_options.index(st.session_state.selected_city) if st.session_state.selected_city in city_options else 0

# --- SIDEBAR REDESIGN ---
st.sidebar.markdown(
    "<div style='padding: 10px 0;'>"
    "<span class='ai-badge'>🏙️ URBANCOOL INTELLIGENCE</span>"
    "<h2 style='margin: 5px 0 0 0; font-size: 30px; font-weight: 800; color: #1E3A8A !important;'>UrbanCool AI</h2>"
    "<p style='color: #64748b; font-size: 13px; margin: 0;'>Combatting Urban Heat Islands</p>"
    "</div>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Navigation simulated selector (with a separate bar for AI Planner Assistant)
st.sidebar.markdown("<p style='font-size: 12px; color:#475569; font-weight:700; text-transform:uppercase;'>System Navigation</p>", unsafe_allow_html=True)
nav_view = st.sidebar.radio(
    "Nav",
    [
        "📊 Heat Intelligence Dashboard", 
        "🧪 Digital Twin Sandbox", 
        "🤖 AI Planner Assistant",
        "🎯 Strategy recommendations", 
        "ℹ️ About Platform"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# City Dropdown selection
st.sidebar.markdown("<p style='font-size: 12px; color:#475569; font-weight:700; text-transform:uppercase;'>Location Workspace</p>", unsafe_allow_html=True)
selected_city = st.sidebar.selectbox(
    "Active City",
    options=city_options,
    index=active_index,
    label_visibility="collapsed"
)

if selected_city != st.session_state.selected_city:
    st.session_state.selected_city = selected_city
    st.rerun()

# Load specific city baseline values
city_baseline = df_cities[df_cities["city"] == selected_city].iloc[0].to_dict()

# Baseline dictionary structure
baseline_features = {
    "green_cover": float(city_baseline.get("baseline_green_cover", 20.0)),
    "building_density": float(city_baseline.get("baseline_building_density", 50.0)),
    "road_density": float(city_baseline.get("baseline_road_density", 45.0)),
    "albedo": float(city_baseline.get("baseline_albedo", 0.15)),
    "humidity": 50.0,
    "wind_speed": 12.0,
    "aqi": 110.0,
    "water_body": float(city_baseline.get("baseline_water_body", 2.0)),
    "solar_radiation": 650.0,
    "surface_moisture": 35.0,
    "elevation": float(city_baseline.get("elevation", 100.0)),
    "pop_density": float(city_baseline.get("pop_density", 5000.0)),
    "climate": city_baseline.get("climate", "inland"),
    "lst": float(city_baseline.get("base_lst", 28.0))
}

# City Landmark representation card
st.sidebar.markdown("<p style='font-size: 12px; color:#475569; font-weight:700; text-transform:uppercase; margin-bottom:5px;'>Landmark View</p>", unsafe_allow_html=True)
landmark_img_path = f"dashboard/assets/{selected_city.lower()}.jpg"
if os.path.exists(landmark_img_path):
    st.sidebar.image(landmark_img_path, use_container_width=True)
else:
    # Beautiful default fallback
    st.sidebar.image("https://images.unsplash.com/photo-1596422846543-c5c6ff183bdf?auto=format&fit=crop&w=400&q=80", use_container_width=True)

# Location Metadata mini details
st.sidebar.markdown(
    f"<div style='margin-top: 5px; font-size: 12px; color: {sec_text_color}; display:flex; justify-content:space-between;'>"
    f"<span>🌐 Coords: {city_baseline['lat']:.2f}°N, {city_baseline['lon']:.2f}°E</span>"
    f"<span>⛰️ Elev: {city_baseline['elevation']}m</span>"
    f"</div>"
    f"<div style='font-size: 12px; color: {sec_text_color}; margin-top: 2px;'>"
    f"👥 Pop Density: {city_baseline['pop_density']} people/km²"
    f"</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Real-time Weather widget
st.sidebar.markdown("<p style='font-size: 12px; color:#475569; font-weight:700; text-transform:uppercase; margin-bottom:5px;'>Ambient Weather</p>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='border: 1px solid {card_border}; background: {card_bg}; padding: 12px; border-radius: 10px;'>"
    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
    f"<span style='font-size:24px; font-weight:800; color:{title_color};'>{baseline_features['lst']:.1f}°C</span>"
    f"<span style='font-size:13px; color:#F59E0B; font-weight:600;'>☀️ Hazy Sun</span>"
    f"</div>"
    f"<div style='font-size:12px; color:{sec_text_color}; margin-top:5px; display:flex; justify-content:space-between;'>"
    f"<span>🌬️ Wind: {baseline_features['wind_speed']:.0f} km/h</span>"
    f"<span>💧 Hum: {baseline_features['humidity']:.0f}%</span>"
    f"</div>"
    f"<div style='font-size:12px; color:{sec_text_color}; margin-top:3px;'>"
    f"😷 AQI Index: <span style='color: {'#EF4444' if baseline_features['aqi'] > 150 else '#10B981'}; font-weight:600;'>{baseline_features['aqi']:.0f}</span>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

# Brand footer
st.sidebar.markdown(
    f"<div class='brand-footer'>"
    f"UrbanCool AI Engine v2.1.0<br>"
    f"© 2026 Smart Infrastructure Board"
    f"</div>",
    unsafe_allow_html=True
)

# Global variables selector from sidebar values or defaults
green_cover = baseline_features["green_cover"]
building_density = baseline_features["building_density"]
road_density = baseline_features["road_density"]
albedo = baseline_features["albedo"]
water_body = baseline_features["water_body"]
aqi = baseline_features["aqi"]
solar_radiation = baseline_features["solar_radiation"]
humidity = baseline_features["humidity"]
wind_speed = baseline_features["wind_speed"]
surface_moisture = baseline_features["surface_moisture"]

# ================= DIGITAL TWIN SANDBOX PARAMETERS =================
if nav_view == "🧪 Digital Twin Sandbox":
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='font-size: 12px; color:#475569; font-weight:700; text-transform:uppercase;'>Sandbox Tuning</p>", unsafe_allow_html=True)
    green_cover = st.sidebar.slider("Green Cover (%)", 0.0, 60.0, baseline_features["green_cover"], 0.5)
    building_density = st.sidebar.slider("Building Density (%)", 10.0, 90.0, baseline_features["building_density"], 0.5)
    road_density = st.sidebar.slider("Road Density (%)", 10.0, 90.0, baseline_features["road_density"], 0.5)
    albedo = st.sidebar.slider("Albedo (Reflectivity)", 0.10, 0.45, baseline_features["albedo"], 0.01)
    water_body = st.sidebar.slider("Water Body Cover (%)", 0.0, 20.0, baseline_features["water_body"], 0.5)
    aqi = st.sidebar.slider("AQI Score", 30.0, 350.0, baseline_features["aqi"], 1.0)
    
modifications = {
    "green_cover": green_cover,
    "building_density": building_density,
    "road_density": road_density,
    "albedo": albedo,
    "water_body": water_body,
    "aqi": aqi,
    "solar_radiation": solar_radiation,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "surface_moisture": surface_moisture
}

# Run simulation calculations
simulator = get_simulator()
sim_results = simulator.run_simulation(baseline_features, modifications)

# ================= HEADER GRID: TITLE AND THEME MODE =================
col_title_left, col_theme_right = st.columns([7.5, 2.5])
with col_title_left:
    st.markdown(
        f"<h1 style='margin-bottom:0; font-size:40px !important;'>Urban Heat Island Intelligence</h1>"
        f"<p style='color:{sec_text_color}; font-size:16px; margin-top:2px;'>Decision support platform for climate-resilient architecture in <b>{selected_city}</b>, India</p>",
        unsafe_allow_html=True
    )
with col_theme_right:
    # Selectbox theme switcher
    theme_select = st.selectbox(
        "Theme Mode Toggle",
        options=["🌙 Dark Mode Theme", "☀️ Light Mode Theme"],
        index=1 if st.session_state.light_mode else 0,
        key="theme_selection_key"
    )
    theme_is_light = (theme_select == "☀️ Light Mode Theme")
    if theme_is_light != st.session_state.light_mode:
        st.session_state.light_mode = theme_is_light
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ================= TOP KPI METRICS BAR =================
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

def render_kpi_card(label, value, baseline, unit="", is_risk=False, reverse=False):
    delta = value - baseline
    delta_str = ""
    if abs(delta) > 0.01:
        sign = "+" if delta > 0 else ""
        if reverse:
            color = "#EF4444" if delta > 0 else "#10B981"
        else:
            color = "#10B981" if delta > 0 else "#EF4444"
        delta_str = f"<span style='color:{color}; font-size:14px; font-weight:700; margin-left:6px;'>{sign}{delta:.1f}{unit}</span>"
        
    val_str = f"{value:.1f}{unit}" if not is_risk else f"{int(value)}/100"
    base_str = f"{baseline:.1f}{unit}" if not is_risk else f"{int(baseline)}"
    
    return f'<div style="background: {card_bg}; border: 1px solid {card_border}; padding: 18px; border-radius: 12px; box-shadow: {card_shadow}; height: 110px; display: flex; flex-direction: column; justify-content: center;"><div style="font-size: 11px; color: {sec_text_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div><div style="display: flex; align-items: baseline; margin-top: 4px;"><span style="font-size: 24px; font-weight: 800; color: {metric_text};">{val_str}</span>{delta_str}</div><div style="font-size: 11px; color: {sec_text_color}; margin-top: 3px;">Base: {base_str}</div></div>'

with kpi_col1:
    st.markdown(render_kpi_card("Predicted Temp (LST)", sim_results["simulated_lst"], sim_results["baseline_lst"], "°C", reverse=True), unsafe_allow_html=True)
with kpi_col2:
    st.markdown(render_kpi_card("Heat Risk Index", sim_results["simulated_risk_score"], sim_results["baseline_risk_score"], is_risk=True, reverse=True), unsafe_allow_html=True)
with kpi_col3:
    st.markdown(render_kpi_card("Urban Green Cover", green_cover, baseline_features["green_cover"], "%"), unsafe_allow_html=True)
with kpi_col4:
    st.markdown(render_kpi_card("Air Quality Index", aqi, baseline_features["aqi"], reverse=True), unsafe_allow_html=True)
with kpi_col5:
    st.markdown(render_kpi_card("Wind Velocity", wind_speed, baseline_features["wind_speed"], " km/h"), unsafe_allow_html=True)
with kpi_col6:
    st.markdown(render_kpi_card("Relative Humidity", humidity, baseline_features["humidity"], "%"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= VIEW 1: DASHBOARD (HEAT MAP & MAIN GRID) =================
if nav_view == "📊 Heat Intelligence Dashboard":
    # 2-Column Middle Layout: Left (Big Map) | Right (Stacked cards: Risk Indicator, then Cooling Interventions)
    col_mid_left, col_mid_right = st.columns([1.6, 1.4])
    
    # Left: Bigger India Urban Heat Map
    with col_mid_left:
        with st.container(border=True):
            st.subheader("India Urban Heat Map")
            st.markdown(f"<p style='font-size: 13px; color: {sec_text_color}; margin-top:-10px; margin-bottom:10px;'>Color represents local Heat Risk category.</p>", unsafe_allow_html=True)
            
            # Map generation
            m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles="CartoDB dark_matter")
            for idx, row in df_cities.iterrows():
                temp = float(row["base_lst"])
                risk_calc = compute_heat_risk(
                    lst=temp,
                    building_density=float(row["baseline_building_density"]),
                    aqi=110.0,
                    humidity=50.0,
                    green_cover=float(row["baseline_green_cover"]),
                    wind_speed=12.0
                )
                
                color = "green"
                if risk_calc["level"] == "Moderate":
                    color = "orange"
                elif risk_calc["level"] == "High":
                    color = "lightred"
                elif risk_calc["level"] == "Extreme":
                    color = "red"
                    
                popup_text = f"<b>{row['city']}</b><br>LST: {temp:.1f}°C<br>Risk: {risk_calc['score']:.0f}"
                icon = folium.Icon(color=color, icon="info-sign")
                if row["city"] == selected_city:
                    icon = folium.Icon(color="purple", icon="star")
                    
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    popup=popup_text,
                    tooltip=row["city"],
                    icon=icon
                ).add_to(m)
                
            # Render bigger map (height 430, width 540)
            map_data = st_folium(m, height=430, width=540, key="dashboard_grid_map")
            
            # Map synchronization
            if map_data and map_data.get("last_object_clicked"):
                c_lat = map_data["last_object_clicked"]["lat"]
                c_lon = map_data["last_object_clicked"]["lng"]
                dists = np.sqrt((df_cities["lat"] - c_lat)**2 + (df_cities["lon"] - c_lon)**2)
                closest_city = df_cities.loc[dists.idxmin(), "city"]
                if closest_city != st.session_state.selected_city:
                    st.session_state.selected_city = closest_city
                    st.rerun()
                    
    # Right: Stacked Cards (One after another)
    with col_mid_right:
        # Stacked Section 1: Heat Risk Assessment
        with st.container(border=True):
            st.subheader("Heat Risk Assessment")
            
            # Semi-circular gauge chart
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sim_results["simulated_risk_score"],
                number = {'font': {'color': metric_text, 'size': 24}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': sec_text_color, 'tickfont': {'size': 10}},
                    'bar': {'color': sim_results["simulated_risk_color"]},
                    'bgcolor': "rgba(255,255,255,0.03)",
                    'borderwidth': 1,
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.08)'},
                        {'range': [35, 55], 'color': 'rgba(251, 191, 36, 0.08)'},
                        {'range': [55, 75], 'color': 'rgba(249, 115, 22, 0.08)'},
                        {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                    ]
                }
            ))
            fig_g.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=130
            )
            st.plotly_chart(fig_g, use_container_width=True)
            
            # Tiny factors list
            st.markdown(
                f"<div style='font-size: 12px; color:{sec_text_color}; text-transform:uppercase; font-weight:700; margin-bottom:5px;'>Microclimate drivers</div>"
                f"<div style='font-size: 13px; display:flex; justify-content:space-between; margin-bottom:2px;'>"
                f"<span>🧱 Building Footprint</span>"
                f"<span>{baseline_features['building_density']:.0f}%</span>"
                f"</div>"
                f"<div style='font-size: 13px; display:flex; justify-content:space-between; margin-bottom:2px;'>"
                f"<span>🌳 Green Space Canopy</span>"
                f"<span>{green_cover:.1f}%</span>"
                f"</div>"
                f"<div style='font-size: 13px; display:flex; justify-content:space-between;'>"
                f"<span>🛣️ Road Footprint</span>"
                f"<span>{baseline_features['road_density']:.0f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        # Stacked Section 2: Cooling Interventions
        with st.container(border=True):
            st.subheader("Cooling Interventions")
            
            # Fetch catalog
            recs_dict = get_cooling_recommendations(baseline_features)
            recs = recs_dict["all_recommendations"][:3]  # top 3
            
            rows_html = []
            for r in recs:
                # Progress bar calculation
                roi_pct = min(100.0, r["roi"] * 1000.0)
                rows_html.append(
                    f"<tr style='border-bottom:1px solid {card_border}; font-size: 12px;'>"
                    f"<td style='padding:6px 0; font-weight:600; color:{text_color};'>{r['name']}</td>"
                    f"<td style='padding:6px 0; color:{sec_text_color};'>₹{r['cost_per_unit_lakhs']:.0f}L</td>"
                    f"<td style='padding:6px 0; text-align:right;'>"
                    f"<div style='width:50px; background:rgba(255,255,255,0.05); height:5px; border-radius:3px; display:inline-block; overflow:hidden;'>"
                    f"<div style='width:{roi_pct:.0f}%; background:#10B981; height:100%;'></div>"
                    f"</div>"
                    f"</td>"
                    f"</tr>"
                )
                
            table_html = f"""
            <table style='width:100%; border-collapse:collapse; margin-top:5px;'>
                <tr style='border-bottom: 1px solid {card_border}; text-align:left; font-size:11px; color:{sec_text_color}; font-weight:700;'>
                    <th style='padding:4px 0;'>STRATEGY</th>
                    <th style='padding:4px 0;'>UNIT COST</th>
                    <th style='padding:4px 0; text-align:right;'>ROI SCALE</th>
                </tr>
                {"".join(rows_html)}
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
    # Bottom Row Layout (Two wider columns since Chat is moved to its own page)
    col_bot_left, col_bot_right = st.columns([1.5, 1.5])
    
    # 4. Digital Twin Simulation
    with col_bot_left:
        with st.container(border=True):
            st.subheader("Digital Twin Simulation")
            
            # Simple before vs after parameter matrix
            sim_rows = [
                ("Green Cover (%)", baseline_features["green_cover"], green_cover),
                ("Building Density (%)", baseline_features["building_density"], building_density),
                ("Road Density (%)", baseline_features["road_density"], road_density),
                ("Albedo Reflectance", baseline_features["albedo"], albedo),
                ("Water Body (%)", baseline_features["water_body"], water_body)
            ]
            
            sim_html_rows = []
            for name, base_v, sim_v in sim_rows:
                diff = sim_v - base_v
                if abs(diff) < 0.001:
                    diff_txt = f"<span style='color:{sec_text_color};'>-</span>"
                else:
                    sign = "+" if diff > 0 else ""
                    color = "#10B981" if (diff > 0 and "Green" in name or "Albedo" in name or "Water" in name) or (diff < 0 and "Building" in name) else "#EF4444"
                    diff_txt = f"<span style='color:{color}; font-weight:600;'>{sign}{diff:.2f}</span>"
                    
                sim_html_rows.append(
                    f"<tr style='border-bottom:1px solid {card_border}; font-size:12px;'>"
                    f"<td style='padding:6px 0; color:{sec_text_color};'>{name}</td>"
                    f"<td style='padding:6px 0; font-weight:600;'>{base_v:.2f}</td>"
                    f"<td style='padding:6px 0; font-weight:600; color:{text_color};'>{sim_v:.2f}</td>"
                    f"<td style='padding:6px 0; text-align:right;'>{diff_txt}</td>"
                    f"</tr>"
                )
                
            sim_table = f"""
            <table style='width:100%; border-collapse:collapse; margin-top:5px;'>
                <tr style='border-bottom:1px solid {card_border}; text-align:left; font-size:11px; color:{sec_text_color}; font-weight:700;'>
                    <th style='padding:4px 0;'>VARIABLE</th>
                    <th style='padding:4px 0;'>BASELINE</th>
                    <th style='padding:4px 0;'>SIMULATED</th>
                    <th style='padding:4px 0; text-align:right;'>DELTA</th>
                </tr>
                {"".join(sim_html_rows)}
            </table>
            """
            st.markdown(sim_table, unsafe_allow_html=True)
            
    # 5. Explainability SHAP
    with col_bot_right:
        with st.container(border=True):
            st.subheader("AI Explainability (SHAP)")
            
            # Generate mini horizontal bar contributions
            shap_w = get_shap_wrapper()
            baseline_df = preprocess_features(baseline_features)
            exp = shap_w.explain(baseline_df)
            
            top_contr = exp["contributions"][:4]  # top 4
            
            fig_s = go.Figure(go.Bar(
                x=[c["shap_value"] for c in top_contr],
                y=[c["display_name"] for c in top_contr],
                orientation='h',
                marker_color=["#EF4444" if c["shap_value"] >= 0 else "#10B981" for c in top_contr],
                text=[f"{c['shap_value']:+.1f}°C" for c in top_contr],
                textposition='inside',
                textfont=dict(color="#ffffff", size=10)
            ))
            
            fig_s.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False, 
                    title=dict(text="Impact (°C)", font=dict(size=10)), 
                    color=sec_text_color, 
                    tickfont=dict(size=10)
                ),
                yaxis=dict(autorange="reversed", color=sec_text_color, tickfont=dict(size=10)),
                margin=dict(l=5, r=5, t=5, b=5),
                height=180
            )
            st.plotly_chart(fig_s, use_container_width=True)

# ================= VIEW 2: DIGITAL TWIN SANDBOX =================
elif nav_view == "🧪 Digital Twin Sandbox":
    # Show side by side dashboard comparing Baseline vs. Simulated
    st.subheader("Digital Twin Comparison Console")
    st.markdown(f"<p style='font-size:14px; color:{sec_text_color}; margin-top:-10px;'>Adjust sliders in the left panel to modify physical variables.</p>", unsafe_allow_html=True)
    
    col_twin_left, col_twin_right = st.columns([1.2, 1.8])
    
    with col_twin_left:
        with st.container(border=True):
            st.subheader("Tuned Parameter Mapping")
            st.markdown(
                f"<div style='font-size:13px; margin-bottom:10px; color:{text_color};'>"
                f"• Green Cover: <b>{green_cover:.1f}%</b> (Base: {baseline_features['green_cover']:.1f}%)<br>"
                f"• Building Density: <b>{building_density:.1f}%</b> (Base: {baseline_features['building_density']:.1f}%)<br>"
                f"• Road Density: <b>{road_density:.1f}%</b> (Base: {baseline_features['road_density']:.1f}%)<br>"
                f"• Albedo Score: <b>{albedo:.2f}</b> (Base: {baseline_features['albedo']:.2f})<br>"
                f"• Water Body: <b>{water_body:.1f}%</b> (Base: {baseline_features['water_body']:.1f}%)<br>"
                f"• AQI Score: <b>{aqi:.0f}</b> (Base: {baseline_features['aqi']:.0f})"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Display summary delta text
            cooling_delta = sim_results["lst_change"]
            if cooling_delta < 0:
                st.success(f"🌱 Temperature reduction of **{abs(cooling_delta):.2f}°C** projected successfully.")
            elif cooling_delta > 0:
                st.error(f"⚠️ Temperature increase of **{cooling_delta:.2f}°C** projected. Reconsider edits.")
            else:
                st.info("No modifications detected.")
                
    with col_twin_right:
        with st.container(border=True):
            st.subheader("Digital Twin Forecast Analysis")
            
            # Semicircular gauges or Plotly indicator
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Indicator(
                mode = "number+delta",
                value = sim_results["simulated_lst"],
                number = {'suffix': "°C", 'font': {'size': 44, 'color': metric_text}},
                delta = {'position': "top", 'reference': sim_results["baseline_lst"], 'valueformat': "+.2f"},
                domain = {'x': [0, 0.45], 'y': [0, 1]},
                title = {'text': "Simulated LST", 'font': {'size': 14, 'color': sec_text_color}}
            ))
            fig_ind.add_trace(go.Indicator(
                mode = "number+delta",
                value = sim_results["simulated_risk_score"],
                number = {'font': {'size': 44, 'color': metric_text}},
                delta = {'position': "top", 'reference': sim_results["baseline_risk_score"], 'valueformat': "+.1f"},
                domain = {'x': [0.55, 1], 'y': [0, 1]},
                title = {'text': "Simulated Heat Risk", 'font': {'size': 14, 'color': sec_text_color}}
            ))
            fig_ind.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=180
            )
            st.plotly_chart(fig_ind, use_container_width=True)

# ================= VIEW 3: DEDICATED AI PLANNER ASSISTANT =================
elif nav_view == "🤖 AI Planner Assistant":
    st.subheader("AI Urban Planner Assistant")
    st.markdown(f"<p style='font-size:14px; color:{sec_text_color}; margin-top:-10px;'>Ask diagnostic queries and receive optimized cooling recommendations for active planning zone.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        # Render the chat log inside a scrollable container
        history_html = []
        for msg in st.session_state.chat_history:
            color = "#818CF8" if msg["role"] == "assistant" else "#10B981"
            prefix = "🤖" if msg["role"] == "assistant" else "👤"
            history_html.append(
                f"<div style='margin-bottom:15px; font-size:14px; background: {sub_bg}; padding: 15px; border-radius: 8px; border: 1px solid {card_border};'>"
                f"<b style='color:{color}; font-size:13px;'>{prefix} {msg['role'].upper()}:</b><br>"
                f"<span style='color:{text_color}; line-height:1.6;'>{msg['content']}</span>"
                f"</div>"
            )
        
        # Display chat log
        st.markdown(
            f"<div style='height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 20px;'>"
            f"{''.join(history_html)}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # User input form
        with st.form(key="dedicated_chat_form", clear_on_submit=True):
            user_msg = st.text_input("Ask the AI Planner...", placeholder="e.g. how to reduce heat risk in Jaipur?")
            chat_submitted = st.form_submit_button("Send Query")
            
        if chat_submitted and user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            
            # Predict reply
            query_lower = user_msg.lower()
            if "jaipur" in query_lower or "desert" in query_lower:
                ai_reply = "Jaipur is in a desert climate zone. LST is heavily driven by solar radiation and low baseline moisture. Prioritize increasing **Albedo** (cool roofs) to reflect sunlight and planting native dry shrubbery."
            elif "delhi" in query_lower or "inland" in query_lower:
                ai_reply = "Delhi has high population and building footprint density. Cool roofs (high-albedo) and urban forestry along transport lanes are the highest ROI strategies to filter AQI and cool pavements."
            elif "mumbai" in query_lower or "coastal" in query_lower:
                ai_reply = "Mumbai is coastal and highly humid. Heat risk is amplified by humidity. Prioritize building density limits, cool roof coating, and protecting coastal wetlands to aid wind-based cooling."
            elif "bengaluru" in query_lower or "bangalore" in query_lower:
                ai_reply = "Bengaluru has a plateau elevation advantage, but rapid building expansions have reduced green canopy. Afforestation and rooftop gardens (green roofs) will maximize local evapotranspiration."
            elif "reduce heat" in query_lower or "cool" in query_lower or "mitigate" in query_lower:
                recs_dict = get_cooling_recommendations(baseline_features)
                top_roi = recs_dict["all_recommendations"][0]["name"]
                ai_reply = f"For **{selected_city}**, the highest ROI intervention is **{top_roi}**. You can run digital twin scenarios to verify the expected LST drop."
            else:
                ai_reply = f"Understood. For the active city ({selected_city}), our Random Forest model indicates that variables like Green Cover and Albedo are primary levers. Consider running digital twin sliders to evaluate cooling ROI."
                
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
            st.rerun()

# ================= VIEW 4: STRATEGY RECOMMENDATIONS =================
elif nav_view == "🎯 Strategy recommendations":
    st.subheader("Urban Heat Mitigation Strategies & ROI Solver")
    st.markdown(f"<p style='font-size:14px; color:{sec_text_color}; margin-top:-10px;'>Mitigation costs are calculated for a standardized 1 sq km planning sector.</p>", unsafe_allow_html=True)
    
    # Get complete list
    rec_results = get_cooling_recommendations(baseline_features)
    all_recs = rec_results["all_recommendations"]
    
    col_rec_in, col_rec_graph = st.columns([1.2, 1.8])
    
    with col_rec_in:
        with st.container(border=True):
            st.subheader("Budget Allocator")
            budget_lakhs = st.number_input(
                "Mitigation Budget (₹ Lakhs)", 
                min_value=0.0, 
                max_value=2000.0, 
                value=250.0, 
                step=50.0
            )
            
            allocated_results = get_cooling_recommendations(baseline_features, budget_lakhs=budget_lakhs)
            allocs = allocated_results["allocated_recommendations"]
            
            st.write(f"💼 **Total Budget**: ₹{budget_lakhs:.1f} Lakhs")
            st.write(f"💸 **Allocated Spend**: ₹{allocated_results['total_cost_lakhs']:.1f} Lakhs")
            st.write(f"💰 **Unspent Funds**: ₹{allocated_results['remaining_budget_lakhs']:.1f} Lakhs")
            
            cooling_reduction = allocated_results["total_cooling_impact"]
            st.markdown(
                f"<div style='border: 1px solid rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.05); padding: 12px; border-radius: 8px; margin-top: 15px; text-align:center;'>"
                f"<h5 style='color: #10B981; margin: 0; font-size:13px;'>Projected Cooling Reduction</h5>"
                f"<h2 style='margin: 5px 0 0 0; color:{text_color};'>-{cooling_reduction:.2f}°C LST</h2>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
    with col_rec_graph:
        with st.container(border=True):
            st.subheader("Cost vs. Cooling Efficiency")
            
            # Plotly Cost vs Impact
            sc_data = []
            for r in all_recs:
                sc_data.append({
                    "Strategy": r["name"],
                    "Unit Cost (₹ Lakhs)": r["cost_per_unit_lakhs"],
                    "Cooling Impact (°C)": r["cooling_per_unit"],
                    "ROI Index": r["roi"] * 10,
                    "Max Scope": r["max_units"]
                })
            df_sc = pd.DataFrame(sc_data)
            
            fig_sc = px.scatter(
                df_sc,
                x="Unit Cost (₹ Lakhs)",
                y="Cooling Impact (°C)",
                size="Max Scope",
                color="ROI Index",
                text="Strategy",
                color_continuous_scale="Viridis",
                size_max=30
            )
            fig_sc.update_traces(textposition='top center')
            fig_sc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color=sec_text_color),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color=sec_text_color),
                margin=dict(l=10, r=10, t=10, b=10),
                height=220
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            
    # Display portfolio allocation details
    if allocs:
        with st.container(border=True):
            st.subheader("Mitigation Portfolio Allocation")
            table_rows = []
            for idx, a in enumerate(allocs):
                table_rows.append({
                    "Rank": idx + 1,
                    "Mitigation Strategy": a["name"],
                    "Allocated Scope": f"{a['units']} {a['unit_name']}",
                    "Cost (₹ Lakhs)": f"₹{a['cost_lakhs']:.1f} Lakhs",
                    "Cooling Drop (°C)": f"-{a['cooling_impact']:.3f}°C"
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ================= VIEW 5: ABOUT PLATFORM =================
else:
    st.subheader("About UrbanCool AI Platform")
    with st.container(border=True):
        st.markdown(
            f"<h4>Evaporative & Albedo Planning Engine</h4>"
            f"<p style='color:{sec_text_color}; font-size:15px; line-height:1.6;'>"
            f"UrbanCool AI uses advanced Random Forest regression algorithms trained on 20,000 environmental combinations "
            f"across 50 diverse Indian geographic locations. The model analyzes thermodynamic surface energy balances "
            f"to calculate how scaling green cover, albedo, and surface moisture helps suppress localized UHI thermal capture.<br><br>"
            f"This decision platform aligns with <b>UN SDG Goal 11 - Sustainable Cities and Communities</b>, assisting urban planners, "
            f"architects, and municipal bodies in evaluating infrastructure portfolios before deployment."
            f"</p>",
            unsafe_allow_html=True
        )
