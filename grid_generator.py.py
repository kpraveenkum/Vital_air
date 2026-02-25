"""
grid_generator.py - Generate grid points for ML heatmap interpolation
Integrates with real API data from data_processor.py and supports both Delhi and Maharashtra
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any

# ========== STATE BOUNDARIES (Synced with main.py and ml_processor.py) ==========
STATE_BOUNDS = {
    'delhi': {
        'name': 'Delhi NCR',
        'lat_min': 28.4, 'lat_max': 28.9,
        'lon_min': 76.8, 'lon_max': 77.3,
        'center': {'lat': 28.6139, 'lon': 77.2090},
        'area_km2': 5500,
        'grid_density': {
            'low': 50,      # 2,500 points (every ~1.1km)
            'medium': 100,  # 10,000 points (every ~550m)
            'high': 200,    # 40,000 points (every ~275m)
            'ultra': 300    # 90,000 points (every ~180m)
        }
    },
    'maharashtra': {
        'name': 'Maharashtra',
        'lat_min': 15.6, 'lat_max': 22.0,
        'lon_min': 72.6, 'lon_max': 80.9,
        'center': {'lat': 19.0760, 'lon': 72.8777},
        'area_km2': 307713,
        'grid_density': {
            'low': 50,      # 2,500 points (every ~14km)
            'medium': 100,  # 10,000 points (every ~7km)
            'high': 150,    # 22,500 points (every ~4.7km)
            'ultra': 200    # 40,000 points (every ~3.5km)
        }
    }
}

# ========== MAJOR CITIES FOR ENHANCED RESOLUTION ==========
# These locations get HIGHER DENSITY grid points for better accuracy
CITY_HOTSPOTS = {
    'delhi': [
        {'name': 'New Delhi', 'lat': 28.6139, 'lon': 77.2090, 'weight': 2.0, 'aqi_factor': 1.3},
        {'name': 'Anand Vihar', 'lat': 28.6468, 'lon': 77.3164, 'weight': 2.5, 'aqi_factor': 1.5},
        {'name': 'ITO', 'lat': 28.6298, 'lon': 77.2423, 'weight': 2.2, 'aqi_factor': 1.4},
        {'name': 'RK Puram', 'lat': 28.5633, 'lon': 77.1769, 'weight': 1.8, 'aqi_factor': 1.2},
        {'name': 'Dwarka', 'lat': 28.5704, 'lon': 77.0653, 'weight': 1.5, 'aqi_factor': 1.1},
        {'name': 'Rohini', 'lat': 28.7344, 'lon': 77.0895, 'weight': 1.5, 'aqi_factor': 1.1},
        {'name': 'Noida', 'lat': 28.5355, 'lon': 77.3910, 'weight': 2.0, 'aqi_factor': 1.3},
        {'name': 'Gurgaon', 'lat': 28.4595, 'lon': 77.0266, 'weight': 1.8, 'aqi_factor': 1.2},
        {'name': 'Faridabad', 'lat': 28.4089, 'lon': 77.3178, 'weight': 1.5, 'aqi_factor': 1.1},
        {'name': 'Ghaziabad', 'lat': 28.6692, 'lon': 77.4538, 'weight': 1.5, 'aqi_factor': 1.2},
    ],
    'maharashtra': [
        {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'weight': 2.5, 'aqi_factor': 1.3},
        {'name': 'Pune', 'lat': 18.5204, 'lon': 73.8567, 'weight': 2.0, 'aqi_factor': 1.2},
        {'name': 'Nagpur', 'lat': 21.1458, 'lon': 79.0882, 'weight': 1.8, 'aqi_factor': 1.1},
        {'name': 'Nashik', 'lat': 19.9975, 'lon': 73.7898, 'weight': 1.5, 'aqi_factor': 1.0},
        {'name': 'Aurangabad', 'lat': 19.8762, 'lon': 75.3433, 'weight': 1.3, 'aqi_factor': 1.0},
        {'name': 'Solapur', 'lat': 17.6599, 'lon': 75.9064, 'weight': 1.2, 'aqi_factor': 0.9},
        {'name': 'Kolhapur', 'lat': 16.6913, 'lon': 74.2449, 'weight': 1.2, 'aqi_factor': 0.9},
        {'name': 'Thane', 'lat': 19.2183, 'lon': 72.9781, 'weight': 1.5, 'aqi_factor': 1.1},
    ]
}

def generate_grid_points(state: str = 'delhi', density: str = 'medium') -> List[Tuple[float, float]]:
    """
    Generate a uniform grid of points for a state
    
    Args:
        state: 'delhi' or 'maharashtra'
        density: 'low', 'medium', 'high', or 'ultra'
    
    Returns:
        List of (lat, lon) tuples
    
    This function is used by ml_processor.py to create base grid points
    """
    if state not in STATE_BOUNDS:
        print(f"⚠️ Unknown state: {state}, defaulting to delhi")
        state = 'delhi'
    
    bounds = STATE_BOUNDS[state]
    steps = bounds['grid_density'].get(density, bounds['grid_density']['medium'])
    
    grid_points = []
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / steps
    lon_step = (bounds['lon_max'] - bounds['lon_min']) / steps
    
    for i in range(steps):
        lat = bounds['lat_min'] + i * lat_step
        for j in range(steps):
            lon = bounds['lon_min'] + j * lon_step
            grid_points.append((round(lat, 4), round(lon, 4)))
    
    print(f"📍 Generated {len(grid_points)} base grid points for {state} at {density} density")
    return grid_points

def generate_adaptive_grid(state: str = 'delhi', base_density: str = 'medium', enhance_cities: bool = True) -> List[Tuple[float, float]]:
    """
    Generate adaptive grid with higher density around major cities
    
    Args:
        state: 'delhi' or 'maharashtra'
        base_density: Base grid density
        enhance_cities: Whether to add extra points around cities
    
    Returns:
        List of (lat, lon) tuples with enhanced resolution
    
    This is the main function used by ml_processor.py for better heatmap quality
    """
    # Start with base grid
    grid_points = generate_grid_points(state, base_density)
    
    if not enhance_cities or state not in CITY_HOTSPOTS:
        return grid_points
    
    bounds = STATE_BOUNDS[state]
    cities = CITY_HOTSPOTS[state]
    
    # Add extra points around each city based on its weight
    total_extra = 0
    for city in cities:
        # Higher weight = more points
        extra_points = int(150 * city['weight'])
        
        for _ in range(extra_points):
            # Random offset within ~3km radius
            lat_offset = (random.random() - 0.5) * 0.05
            lon_offset = (random.random() - 0.5) * 0.05
            
            new_lat = city['lat'] + lat_offset
            new_lon = city['lon'] + lon_offset
            
            # Ensure within state bounds
            if (bounds['lat_min'] <= new_lat <= bounds['lat_max'] and
                bounds['lon_min'] <= new_lon <= bounds['lon_max']):
                grid_points.append((round(new_lat, 4), round(new_lon, 4)))
                total_extra += 1
    
    print(f"📍 Added {total_extra} enhanced grid points around {len(cities)} cities for {state}")
    print(f"📍 Total grid points: {len(grid_points)}")
    
    return grid_points

def generate_weighted_grid_points(state: str = 'delhi', density: str = 'medium', 
                                  sensor_locations: List[Dict[str, Any]] = None) -> List[Tuple[float, float]]:
    """
    Generate grid points with weights based on sensor density
    
    Args:
        state: 'delhi' or 'maharashtra'
        density: Base grid density
        sensor_locations: List of sensor locations from data_processor.py
    
    Returns:
        List of (lat, lon) tuples with higher density where sensors are clustered
    
    This function uses real sensor locations to determine where to add more points
    """
    # Start with adaptive grid
    grid_points = generate_adaptive_grid(state, density)
    
    if not sensor_locations:
        return grid_points
    
    bounds = STATE_BOUNDS[state]
    
    # Count sensors per area to identify clusters
    sensor_grid = {}
    for sensor in sensor_locations:
        lat_key = round(sensor['lat'], 1)
        lon_key = round(sensor['lon'], 1)
        key = (lat_key, lon_key)
        sensor_grid[key] = sensor_grid.get(key, 0) + 1
    
    # Add extra points in high-density sensor areas
    extra_added = 0
    for (lat_key, lon_key), count in sensor_grid.items():
        if count >= 2:  # Area with multiple sensors
            # Add extra points proportional to sensor count
            extra_points = count * 20
            
            for _ in range(extra_points):
                lat = lat_key + (random.random() - 0.5) * 0.2
                lon = lon_key + (random.random() - 0.5) * 0.2
                
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):
                    grid_points.append((round(lat, 4), round(lon, 4)))
                    extra_added += 1
    
    if extra_added > 0:
        print(f"📍 Added {extra_added} grid points based on sensor density")
    
    return list(set(grid_points))  # Remove duplicates

def get_grid_statistics(state: str, density: str = 'medium') -> Dict[str, Any]:
    """
    Get statistics about the generated grid
    
    Returns:
        Dictionary with grid statistics for debugging
    """
    if state not in STATE_BOUNDS:
        return {"error": "Unknown state"}
    
    bounds = STATE_BOUNDS[state]
    steps = bounds['grid_density'].get(density, bounds['grid_density']['medium'])
    
    lat_range = bounds['lat_max'] - bounds['lat_min']
    lon_range = bounds['lon_max'] - bounds['lon_min']
    
    lat_step = lat_range / steps
    lon_step = lon_range / steps
    
    # Approximate distance in km
    lat_dist_km = lat_step * 111  # 1° ≈ 111 km
    lon_dist_km = lon_step * 111 * math.cos(math.radians((bounds['lat_min'] + bounds['lat_max']) / 2))
    
    return {
        'state': state,
        'density': density,
        'grid_points': steps * steps,
        'resolution_km': {
            'lat': round(lat_dist_km, 2),
            'lon': round(lon_dist_km, 2),
            'avg': round((lat_dist_km + lon_dist_km) / 2, 2)
        },
        'bounds': bounds
    }

def generate_grid_for_ml_processor(state: str = 'delhi', use_enhanced: bool = True) -> List[Tuple[float, float]]:
    """
    Main function called by ml_processor.py to generate grid points
    
    This is the recommended function to use in your ML pipeline
    """
    if use_enhanced:
        # Use adaptive grid with city hotspots for better accuracy
        return generate_adaptive_grid(state, base_density='medium', enhance_cities=True)
    else:
        # Use simple uniform grid
        return generate_grid_points(state, density='medium')

def generate_multi_state_grid() -> Dict[str, List[Tuple[float, float]]]:
    """
    Generate grids for all supported states at once
    
    Returns:
        Dictionary with state names as keys and grid points as values
    """
    grids = {}
    for state in STATE_BOUNDS.keys():
        grids[state] = generate_grid_for_ml_processor(state)
    return grids

def optimize_grid_for_lambda(state: str = 'delhi', max_points: int = 10000) -> List[Tuple[float, float]]:
    """
    Generate grid optimized for AWS Lambda (memory and time constraints)
    
    Args:
        state: 'delhi' or 'maharashtra'
        max_points: Maximum number of grid points (Lambda memory limit)
    
    Returns:
        Optimized grid points list
    """
    bounds = STATE_BOUNDS[state]
    
    # Calculate required density to stay under max_points
    total_area_points = (bounds['lat_max'] - bounds['lat_min']) * (bounds['lon_max'] - bounds['lon_min'])
    target_density = math.sqrt(max_points / total_area_points)
    
    steps = int(target_density * 100)  # Scale appropriately
    
    grid_points = []
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / steps
    lon_step = (bounds['lon_max'] - bounds['lon_min']) / steps
    
    for i in range(steps):
        lat = bounds['lat_min'] + i * lat_step
        for j in range(steps):
            lon = bounds['lon_min'] + j * lon_step
            grid_points.append((round(lat, 4), round(lon, 4)))
    
    print(f"📍 Generated {len(grid_points)} optimized grid points for Lambda")
    return grid_points

# ========== TESTING AND DEBUGGING ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Grid Generator with Real API Data Integration")
    print("=" * 60)
    
    # Test Delhi grid
    print("\n📊 Testing Delhi grid generation:")
    delhi_grid = generate_grid_for_ml_processor('delhi')
    stats = get_grid_statistics('delhi', 'medium')
    print(f"   Total points: {len(delhi_grid)}")
    print(f"   Resolution: {stats['resolution_km']['avg']}km")
    print(f"   Sample points: {delhi_grid[:3]}")
    
    # Test Maharashtra grid
    print("\n📊 Testing Maharashtra grid generation:")
    maha_grid = generate_grid_for_ml_processor('maharashtra')
    stats = get_grid_statistics('maharashtra', 'medium')
    print(f"   Total points: {len(maha_grid)}")
    print(f"   Resolution: {stats['resolution_km']['avg']}km")
    print(f"   Sample points: {maha_grid[:3]}")
    
    # Test with mock sensor data (simulating data_processor.py output)
    print("\n📊 Testing sensor-density based grid:")
    mock_sensors = [
        {'lat': 28.6139, 'lon': 77.2090, 'value': 180},
        {'lat': 28.6468, 'lon': 77.3164, 'value': 220},
        {'lat': 28.6298, 'lon': 77.2423, 'value': 195},
        {'lat': 28.5633, 'lon': 77.1769, 'value': 145},
    ]
    sensor_grid = generate_weighted_grid_points('delhi', 'medium', mock_sensors)
    print(f"   Points with sensor enhancement: {len(sensor_grid)}")
    
    # Test Lambda-optimized grid
    print("\n📊 Testing Lambda-optimized grid:")
    lambda_grid = optimize_grid_for_lambda('delhi', max_points=5000)
    print(f"   Lambda-optimized points: {len(lambda_grid)}")
    
    print("\n✅ Grid Generator tests complete!")