def get_cooling_recommendations(current_features, budget_lakhs=None):
    """
    Evaluates current urban parameters and returns a list of cooling recommendations
    sorted by ROI (Cooling per Lakh Rupees). If a budget is specified, it allocates
    funds to the highest ROI options.
    """
    green_cover = current_features.get("green_cover", 20.0)
    building_density = current_features.get("building_density", 50.0)
    road_density = current_features.get("road_density", 40.0)
    water_body = current_features.get("water_body", 2.0)
    albedo = current_features.get("albedo", 0.15)
    
    recommendations = []
    
    # 1. Urban Forestry
    if green_cover < 50.0:
        max_gc_inc = 50.0 - green_cover
        # Unit: 1% increase in green cover
        # Cost: ₹15 Lakhs (~1,000 trees planted and maintained per sq km)
        # Cooling impact: -0.20 °C
        recommendations.append({
            "key": "urban_forestry",
            "name": "Urban Forestry & Tree Planting",
            "description": "Plant native shade trees in parks, open spaces, and along roadsides to boost green cover and evapotranspiration.",
            "unit": "% increase in Green Cover",
            "unit_cost_desc": "₹15 Lakhs per 1% increase (~1,000 trees)",
            "max_units": round(max_gc_inc, 1),
            "cost_per_unit_lakhs": 15.0,
            "cooling_per_unit": 0.20,
            "roi": round(0.20 / 15.0, 5)  # cooling in °C per lakh
        })
        
    # 2. Cool Roof Coatings
    if building_density > 20.0 and albedo < 0.40:
        max_albedo_inc = 0.40 - albedo
        # Unit: +0.05 increase in albedo (which reflects solar heat)
        # Cost: ₹150 Lakhs to apply high albedo coating on 50,000 sqm roof space (10% of building footprint in dense 1 sq km area)
        # Cooling impact: 0.60 °C (formula coefficient mapping albedo to LST)
        recommendations.append({
            "key": "cool_roofs",
            "name": "Cool Roof Coatings (High-Albedo)",
            "description": "Apply high-reflectivity elastomeric paint on concrete building rooftops to reflect solar radiation.",
            "unit": "+0.05 increase in Albedo",
            "unit_cost_desc": "₹150 Lakhs per +0.05 albedo increase",
            "max_units": round(max_albedo_inc / 0.05, 1),
            "cost_per_unit_lakhs": 150.0,
            "cooling_per_unit": 0.60,
            "roi": round(0.60 / 150.0, 5)
        })
        
    # 3. Green Roofs
    if building_density > 40.0 and green_cover < 45.0:
        max_green_roofs = min(10.0, 50.0 - green_cover)
        # Unit: 1% building rooftop greening (adds 0.5% green cover)
        # Cost: ₹100 Lakhs for 5,000 sqm of intensive/extensive rooftop vegetation
        # Cooling impact: 0.12 °C
        recommendations.append({
            "key": "green_roofs",
            "name": "Intensive/Extensive Green Roofs",
            "description": "Establish vegetation and micro-gardens on building rooftops. Reduces building heat gain and adds green cover.",
            "unit": "1% rooftop area greened",
            "unit_cost_desc": "₹100 Lakhs per 1% rooftop greening (~5,000 m²)",
            "max_units": round(max_green_roofs, 1),
            "cost_per_unit_lakhs": 100.0,
            "cooling_per_unit": 0.12,
            "roi": round(0.12 / 100.0, 5)
        })
        
    # 4. Cool & Permeable Pavements
    if road_density > 20.0:
        max_pav_inc = min(15.0, road_density / 2.0)
        # Unit: 1% road area paved with cool materials
        # Cost: ₹40 Lakhs to resurface 4,000 sqm of pavement/road shoulder with cool concrete block
        # Cooling impact: 0.05 °C
        recommendations.append({
            "key": "cool_pavements",
            "name": "Cool & Permeable Pavements",
            "description": "Replace asphalt with light-colored, porous concrete blocks that reflect heat and absorb moisture.",
            "unit": "1% road area resurfaced",
            "unit_cost_desc": "₹40 Lakhs per 1% road area (~4,000 m²)",
            "max_units": round(max_pav_inc, 1),
            "cost_per_unit_lakhs": 40.0,
            "cooling_per_unit": 0.05,
            "roi": round(0.05 / 40.0, 5)
        })
        
    # 5. Wetland Restoration
    if water_body < 15.0:
        max_water_inc = 15.0 - water_body
        # Unit: 1% increase in water body area
        # Cost: ₹150 Lakhs to restore 10,000 sqm of local ponds/wetland marsh
        # Cooling impact: 0.15 °C
        recommendations.append({
            "key": "wetland_restoration",
            "name": "Urban Wetland & Pond Restoration",
            "description": "Dredge, clean, and restore urban ponds and wetlands to maximize localized evaporative cooling.",
            "unit": "% increase in Water Body cover",
            "unit_cost_desc": "₹150 Lakhs per 1% water body increase (~10,000 m²)",
            "max_units": round(max_water_inc, 1),
            "cost_per_unit_lakhs": 150.0,
            "cooling_per_unit": 0.15,
            "roi": round(0.15 / 150.0, 5)
        })
        
    # Sort recommendations by ROI descending
    recommendations = sorted(recommendations, key=lambda x: x["roi"], reverse=True)
    
    allocated = []
    total_cost_lakhs = 0.0
    total_cooling = 0.0
    
    if budget_lakhs is not None:
        remaining_budget = float(budget_lakhs)
        for rec in recommendations:
            if remaining_budget <= 0.0:
                break
            
            cost_per_unit = rec["cost_per_unit_lakhs"]
            max_units = rec["max_units"]
            
            units_affordable = remaining_budget / cost_per_unit
            units_to_buy = min(max_units, units_affordable)
            
            if units_to_buy > 0.0:
                cost = units_to_buy * cost_per_unit
                cooling = units_to_buy * rec["cooling_per_unit"]
                
                allocated.append({
                    "key": rec["key"],
                    "name": rec["name"],
                    "units": round(units_to_buy, 2),
                    "unit_name": rec["unit"],
                    "cost_lakhs": round(cost, 2),
                    "cooling_impact": round(cooling, 3)
                })
                
                total_cost_lakhs += cost
                total_cooling += cooling
                remaining_budget -= cost
                
    return {
        "all_recommendations": recommendations,
        "allocated_recommendations": allocated,
        "total_cost_lakhs": round(total_cost_lakhs, 2),
        "total_cooling_impact": round(total_cooling, 3),
        "remaining_budget_lakhs": round(remaining_budget if budget_lakhs is not None else 0.0, 2)
    }
