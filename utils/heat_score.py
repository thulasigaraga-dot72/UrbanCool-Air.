def compute_heat_risk(lst, building_density, aqi, humidity, green_cover, wind_speed):
    """
    Computes the Heat Risk Index (0-100) based on weighted physical factors:
    - Land Surface Temperature (LST)
    - Building Density
    - AQI
    - Humidity
    - Green Cover (negative contribution)
    - Wind Speed (negative contribution)
    """
    # Normalize features to 0-100 scale for equation consistency
    # LST range assumed 10°C to 45°C
    lst_n = min(max((lst - 10.0) / 35.0, 0.0), 1.0) * 100.0
    bd_n = min(max(building_density, 0.0), 100.0)
    # AQI range assumed 0 to 350
    aqi_n = min(max(aqi / 350.0, 0.0), 1.0) * 100.0
    hum_n = min(max(humidity, 0.0), 100.0)
    # Green cover range assumed 0% to 60%
    gc_n = min(max(green_cover / 60.0, 0.0), 1.0) * 100.0
    # Wind speed range assumed 2 to 25 km/h
    ws_n = min(max((wind_speed - 2.0) / 23.0, 0.0), 1.0) * 100.0
    
    # Weighted calculation
    # LST (45%), Building Density (20%), AQI (15%), Humidity (10%), Green Cover (-20%), Wind Speed (-5%)
    score = (0.45 * lst_n) + (0.20 * bd_n) + (0.15 * aqi_n) + (0.10 * hum_n) - (0.20 * gc_n) - (0.05 * ws_n)
    
    # Scale score to 0 - 100 (Max positive sum is 90, Min negative sum is -25, range = 115)
    normalized_score = (score + 25.0) * (100.0 / 115.0)
    normalized_score = round(min(max(normalized_score, 0.0), 100.0), 2)
    
    # Categorization
    if normalized_score < 35:
        category = "Low Risk / Cool"
        color = "#10B981"  # Emerald Green
        level = "Low"
    elif normalized_score < 55:
        category = "Moderate Risk"
        color = "#FBBF24"  # Amber Yellow
        level = "Moderate"
    elif normalized_score < 75:
        category = "High Risk"
        color = "#F97316"  # Orange
        level = "High"
    else:
        category = "Extreme Risk"
        color = "#EF4444"  # Red
        level = "Extreme"
        
    return {
        "score": normalized_score,
        "category": category,
        "color": color,
        "level": level
    }
