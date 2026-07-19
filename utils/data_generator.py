import os
import numpy as np
import pandas as pd

def generate_data():
    print("Generating locations database...")
    # List of 50 Indian cities/regions with geographic coordinates, approximate baseline elevation,
    # approximate population density (people/sq km), and climate baseline classification.
    cities_data = [
        {"city": "Delhi", "lat": 28.6139, "lon": 77.2090, "elevation": 213, "pop_density": 11320, "climate": "inland", "base_lst": 27.5},
        {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777, "elevation": 14, "pop_density": 21000, "climate": "coastal", "base_lst": 25.0},
        {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639, "elevation": 9, "pop_density": 24000, "climate": "coastal", "base_lst": 25.5},
        {"city": "Chennai", "lat": 13.0827, "lon": 80.2707, "elevation": 6, "pop_density": 26500, "climate": "coastal", "base_lst": 26.0},
        {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "elevation": 920, "pop_density": 4380, "climate": "plateau", "base_lst": 22.0},
        {"city": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elevation": 542, "pop_density": 18490, "climate": "plateau", "base_lst": 24.0},
        {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873, "elevation": 431, "pop_density": 6500, "climate": "desert", "base_lst": 29.0},
        {"city": "Srinagar", "lat": 34.0837, "lon": 74.7973, "elevation": 1585, "pop_density": 4000, "climate": "mountain", "base_lst": 12.0},
        {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "elevation": 53, "pop_density": 12000, "climate": "inland", "base_lst": 28.0},
        {"city": "Guwahati", "lat": 26.1445, "lon": 91.7362, "elevation": 55, "pop_density": 4000, "climate": "coastal_plain", "base_lst": 23.5},
        {"city": "Pune", "lat": 18.5204, "lon": 73.8567, "elevation": 560, "pop_density": 5600, "climate": "plateau", "base_lst": 23.0},
        {"city": "Lucknow", "lat": 26.8467, "lon": 80.9462, "elevation": 123, "pop_density": 1815, "climate": "inland", "base_lst": 26.5},
        {"city": "Patna", "lat": 25.5941, "lon": 85.1376, "elevation": 53, "pop_density": 1820, "climate": "inland", "base_lst": 26.0},
        {"city": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "elevation": 45, "pop_density": 2700, "climate": "coastal", "base_lst": 25.0},
        {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "elevation": 321, "pop_density": 9250, "climate": "inland", "base_lst": 25.0},
        {"city": "Bhopal", "lat": 23.2599, "lon": 77.4126, "elevation": 427, "pop_density": 3890, "climate": "plateau", "base_lst": 25.0},
        {"city": "Dehradun", "lat": 30.3165, "lon": 78.0322, "elevation": 640, "pop_density": 2000, "climate": "mountain", "base_lst": 18.0},
        {"city": "Shimla", "lat": 31.1048, "lon": 77.1734, "elevation": 2206, "pop_density": 2200, "climate": "mountain", "base_lst": 10.0},
        {"city": "Kochi", "lat": 9.9312, "lon": 76.2673, "elevation": 2, "pop_density": 6340, "climate": "coastal", "base_lst": 24.5},
        {"city": "Coimbatore", "lat": 11.0168, "lon": 76.9558, "elevation": 411, "pop_density": 3100, "climate": "plateau", "base_lst": 24.0},
        {"city": "Nagpur", "lat": 21.1458, "lon": 79.0882, "elevation": 310, "pop_density": 4700, "climate": "inland", "base_lst": 27.0},
        {"city": "Indore", "lat": 22.7196, "lon": 75.8577, "elevation": 553, "pop_density": 9700, "climate": "plateau", "base_lst": 24.5},
        {"city": "Ranchi", "lat": 23.3441, "lon": 85.3096, "elevation": 651, "pop_density": 1000, "climate": "plateau", "base_lst": 22.0},
        {"city": "Raipur", "lat": 21.2514, "lon": 81.6296, "elevation": 298, "pop_density": 3100, "climate": "inland", "base_lst": 26.0},
        {"city": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "elevation": 45, "pop_density": 2100, "climate": "coastal", "base_lst": 25.5},
        {"city": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366, "elevation": 10, "pop_density": 4450, "climate": "coastal", "base_lst": 24.0},
        {"city": "Madurai", "lat": 9.9252, "lon": 78.1198, "elevation": 101, "pop_density": 8200, "climate": "inland", "base_lst": 26.5},
        {"city": "Varanasi", "lat": 25.3176, "lon": 82.9739, "elevation": 81, "pop_density": 9700, "climate": "inland", "base_lst": 26.5},
        {"city": "Amritsar", "lat": 31.6340, "lon": 74.8723, "elevation": 234, "pop_density": 6400, "climate": "inland", "base_lst": 26.0},
        {"city": "Jammu", "lat": 32.7266, "lon": 74.8570, "elevation": 327, "pop_density": 2100, "climate": "mountain", "base_lst": 20.0},
        {"city": "Leh", "lat": 34.1526, "lon": 77.5771, "elevation": 3500, "pop_density": 300, "climate": "mountain", "base_lst": 5.0},
        {"city": "Panaji", "lat": 15.4909, "lon": 73.8278, "elevation": 7, "pop_density": 1200, "climate": "coastal", "base_lst": 24.5},
        {"city": "Jodhpur", "lat": 26.2389, "lon": 73.0243, "elevation": 231, "pop_density": 3400, "climate": "desert", "base_lst": 29.5},
        {"city": "Udaipur", "lat": 24.5854, "lon": 73.7125, "elevation": 598, "pop_density": 2400, "climate": "desert", "base_lst": 27.5},
        {"city": "Shillong", "lat": 25.5788, "lon": 91.8831, "elevation": 1525, "pop_density": 2900, "climate": "mountain", "base_lst": 14.0},
        {"city": "Imphal", "lat": 24.8170, "lon": 93.9368, "elevation": 786, "pop_density": 1100, "climate": "mountain", "base_lst": 18.0},
        {"city": "Agartala", "lat": 23.8315, "lon": 91.2868, "elevation": 15, "pop_density": 5200, "climate": "coastal_plain", "base_lst": 24.0},
        {"city": "Aizawl", "lat": 23.7307, "lon": 92.7173, "elevation": 1132, "pop_density": 2300, "climate": "mountain", "base_lst": 16.0},
        {"city": "Kohima", "lat": 25.6751, "lon": 94.1086, "elevation": 1444, "pop_density": 2100, "climate": "mountain", "base_lst": 14.5},
        {"city": "Itanagar", "lat": 27.0844, "lon": 93.6053, "elevation": 440, "pop_density": 600, "climate": "mountain", "base_lst": 18.5},
        {"city": "Gangtok", "lat": 27.3314, "lon": 88.6138, "elevation": 1650, "pop_density": 2500, "climate": "mountain", "base_lst": 13.0},
        {"city": "Port Blair", "lat": 11.6234, "lon": 92.7265, "elevation": 16, "pop_density": 1200, "climate": "coastal", "base_lst": 24.5},
        {"city": "Kavaratti", "lat": 10.5667, "lon": 72.6417, "elevation": 1, "pop_density": 3300, "climate": "coastal", "base_lst": 24.5},
        {"city": "Puducherry", "lat": 11.9416, "lon": 79.8083, "elevation": 3, "pop_density": 3200, "climate": "coastal", "base_lst": 25.5},
        {"city": "Surat", "lat": 21.1702, "lon": 72.8311, "elevation": 13, "pop_density": 13600, "climate": "coastal", "base_lst": 26.5},
        {"city": "Vadodara", "lat": 22.3072, "lon": 73.1812, "elevation": 39, "pop_density": 8500, "climate": "inland", "base_lst": 27.0},
        {"city": "Rajkot", "lat": 22.3039, "lon": 70.8022, "elevation": 134, "pop_density": 5300, "climate": "inland", "base_lst": 27.5},
        {"city": "Gwalior", "lat": 26.2183, "lon": 78.1828, "elevation": 197, "pop_density": 3800, "climate": "inland", "base_lst": 27.0},
        {"city": "Jabalpur", "lat": 23.1815, "lon": 79.9864, "elevation": 411, "pop_density": 3500, "climate": "plateau", "base_lst": 25.5},
        {"city": "Prayagraj", "lat": 25.4358, "lon": 81.8463, "elevation": 98, "pop_density": 5000, "climate": "inland", "base_lst": 26.5}
    ]

    # Create the baseline locations DataFrame and save
    df_cities = pd.DataFrame(cities_data)
    
    # Generate baseline green cover (from 10% to 35%) and baseline albedo for the location itself
    np.random.seed(42)
    df_cities["baseline_green_cover"] = np.random.uniform(10.0, 35.0, len(df_cities))
    df_cities["baseline_albedo"] = np.random.uniform(0.12, 0.22, len(df_cities))
    df_cities["baseline_building_density"] = np.random.uniform(40.0, 80.0, len(df_cities))
    df_cities["baseline_road_density"] = np.random.uniform(30.0, 60.0, len(df_cities))
    df_cities["baseline_water_body"] = np.random.uniform(1.0, 10.0, len(df_cities))
    
    os.makedirs("data", exist_ok=True)
    df_cities.to_csv("data/locations.csv", index=False)
    print("Locations saved to data/locations.csv")

    print("Generating environmental scenarios dataset (20,000 samples)...")
    scenarios_per_city = 400
    records = []

    for idx, row in df_cities.iterrows():
        # Generate 400 variations for this city
        for _ in range(scenarios_per_city):
            green_cover = np.random.uniform(0.0, 60.0)
            building_density = np.random.uniform(10.0, 90.0)
            road_density = np.random.uniform(10.0, 90.0)
            albedo = np.random.uniform(0.1, 0.45)
            humidity = np.random.uniform(30.0, 90.0)
            wind_speed = np.random.uniform(2.0, 25.0)
            aqi = np.random.uniform(30.0, 350.0)
            water_body = np.random.uniform(0.0, 20.0)
            solar_rad = np.random.uniform(200.0, 1000.0)
            surface_moisture = np.random.uniform(5.0, 80.0)
            
            # Target LST calculation formula:
            # LST = base_lst + 0.015*solar_rad + 0.18*building_density + 0.10*road_density + 0.01*pop_density/1000
            #       - 0.20*green_cover - 0.15*water_body - 12.0*albedo - 0.15*wind_speed - 0.05*humidity
            #       - 0.003*elevation + 0.02*aqi - 0.10*surface_moisture + noise
            
            noise = np.random.normal(0.0, 0.5)
            
            lst = (
                row["base_lst"]
                + 0.015 * solar_rad
                + 0.18 * building_density
                + 0.10 * road_density
                + 0.01 * (row["pop_density"] / 1000.0)
                - 0.20 * green_cover
                - 0.15 * water_body
                - 12.0 * albedo
                - 0.15 * wind_speed
                - 0.05 * humidity
                - 0.003 * row["elevation"]
                + 0.02 * aqi
                - 0.10 * surface_moisture
                + noise
            )
            
            records.append({
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "elevation": row["elevation"],
                "pop_density": row["pop_density"],
                "climate": row["climate"],
                "green_cover": green_cover,
                "building_density": building_density,
                "road_density": road_density,
                "albedo": albedo,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "aqi": aqi,
                "water_body": water_body,
                "solar_radiation": solar_rad,
                "surface_moisture": surface_moisture,
                "lst": lst
            })
            
    df_dataset = pd.DataFrame(records)
    df_dataset.to_csv("data/dataset.csv", index=False)
    print("Scenarios dataset saved to data/dataset.csv")
    print(f"Total samples generated: {len(df_dataset)}")

if __name__ == "__main__":
    generate_data()
