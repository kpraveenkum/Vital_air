"""
idw.py - Core interpolation algorithms for heatmap generation
"""

import math
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371  # Earth's radius
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_idw_single(target_lat, target_lon, data_points, power=2, max_points=20):
    """
    Standard IDW for a single point
    
    Formula: Z = Σ(zi / di²) / Σ(1 / di²)
    """
    if not data_points:
        return None
    
    # Calculate distances and sort
    points_with_dist = []
    for point in data_points:
        dist = haversine_distance(target_lat, target_lon, point['lat'], point['lon'])
        points_with_dist.append({
            'value': point['value'],
            'weight': point.get('weight', 1.0),
            'dist': dist
        })
    
    # Sort by distance
    points_with_dist.sort(key=lambda x: x['dist'])
    
    # Take closest N points
    closest = points_with_dist[:max_points]
    
    # Calculate weighted sum
    weighted_sum = 0
    total_weight = 0
    
    for point in closest:
        dist = point['dist']
        
        if dist < 0.1:  # Extremely close
            return point['value']
        
        # Weight = 1 / distance²
        dist_weight = 1.0 / (dist ** power)
        combined_weight = dist_weight * point['weight']
        
        weighted_sum += combined_weight * point['value']
        total_weight += combined_weight
    
    if total_weight == 0:
        return None
    
    return weighted_sum / total_weight

def calculate_idw_batch(grid_points, data_points, power=2):
    """Calculate IDW for multiple grid points"""
    predictions = []
    
    for lat, lon in grid_points:
        value = calculate_idw_single(lat, lon, data_points, power)
        if value is not None:
            predictions.append(value)
        else:
            predictions.append(None)
    
    return predictions

def calculate_rbf_single(target_lat, target_lon, data_points, epsilon=1.0):
    """
    Radial Basis Function interpolation
    Creates smoother heatmaps than IDW
    
    Formula: φ(r) = exp(-(εr)²)
    """
    if len(data_points) < 3:
        return calculate_idw_single(target_lat, target_lon, data_points)
    
    try:
        # Extract points and values
        points = []
        values = []
        weights = []
        
        for point in data_points:
            points.append((point['lat'], point['lon']))
            values.append(point['value'])
            weights.append(point.get('weight', 1.0))
        
        # Calculate distances
        distances = []
        for s_lat, s_lon in points:
            dist = haversine_distance(target_lat, target_lon, s_lat, s_lon)
            distances.append(dist)
        
        # Gaussian RBF kernel
        rbf_weights = np.exp(-(np.array(distances) ** 2) / (2 * epsilon ** 2))
        
        # Apply source weights
        combined_weights = rbf_weights * np.array(weights)
        
        # Handle near-zero case
        if np.sum(combined_weights) < 1e-10:
            return calculate_idw_single(target_lat, target_lon, data_points)
        
        # Weighted sum
        prediction = np.sum(combined_weights * values) / np.sum(combined_weights)
        return float(prediction)
        
    except Exception as e:
        print(f"RBF error: {e}, falling back to IDW")
        return calculate_idw_single(target_lat, target_lon, data_points)

def calculate_idw_with_temporal(target_lat, target_lon, data_points, current_time=None):
    """
    Enhanced IDW with temporal weighting
    Newer data points have higher weight
    """
    if not data_points:
        return None
    
    if current_time is None:
        current_time = time.time()
    
    # Prepare points with temporal weights
    enhanced_points = []
    for point in data_points:
        # Calculate temporal weight (decay over time)
        if 'timestamp' in point:
            age_hours = (current_time - point['timestamp']) / 3600
            temporal_weight = max(0.3, 1.0 - (age_hours / 72))  # Decay over 72h
        else:
            temporal_weight = 0.7  # Default for points without timestamp
        
        enhanced_points.append({
            'lat': point['lat'],
            'lon': point['lon'],
            'value': point['value'],
            'weight': point.get('weight', 1.0) * temporal_weight
        })
    
    # Use standard IDW with enhanced weights
    return calculate_idw_single(target_lat, target_lon, enhanced_points)

def kriging_simple(target_lat, target_lon, data_points):
    """
    Simplified kriging-like interpolation
    Uses variogram model for spatial correlation
    """
    if len(data_points) < 5:
        return calculate_idw_single(target_lat, target_lon, data_points)
    
    try:
        # Extract values
        points = [(p['lat'], p['lon']) for p in data_points]
        values = [p['value'] for p in data_points]
        
        # Calculate experimental variogram
        distances = []
        semivariances = []
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = haversine_distance(points[i][0], points[i][1], 
                                         points[j][0], points[j][1])
                if dist > 0:
                    distances.append(dist)
                    semivariance = 0.5 * ((values[i] - values[j]) ** 2)
                    semivariances.append(semivariance)
        
        if not distances:
            return calculate_idw_single(target_lat, target_lon, data_points)
        
        # Simple linear variogram model
        # In practice, you'd fit a proper model (spherical, exponential, etc.)
        weights = []
        weighted_sum = 0
        total_weight = 0
        
        for i, point in enumerate(data_points):
            dist = haversine_distance(target_lat, target_lon, point['lat'], point['lon'])
            
            if dist < 0.1:
                return point['value']
            
            # Simplified kriging weight (inverse distance with variogram influence)
            weight = 1.0 / (dist ** 1.5)  # Modified power
            weights.append(weight)
            weighted_sum += weight * point['value']
            total_weight += weight
        
        return weighted_sum / total_weight
        
    except Exception as e:
        print(f"Kriging error: {e}, falling back to IDW")
        return calculate_idw_single(target_lat, target_lon, data_points)

# For testing
if __name__ == "__main__":
    print("🧪 Testing IDW algorithms...")
    
    # Test data
    data_points = [
        {'lat': 28.6139, 'lon': 77.2090, 'value': 180, 'weight': 1.0},
        {'lat': 28.5355, 'lon': 77.3910, 'value': 165, 'weight': 1.0},
        {'lat': 28.4595, 'lon': 77.0266, 'value': 155, 'weight': 1.0},
    ]
    
    test_point = (28.6000, 77.2000)
    
    # Test IDW
    idw_result = calculate_idw_single(test_point[0], test_point[1], data_points)
    print(f"IDW result: {idw_result:.1f}")
    
    # Test RBF
    rbf_result = calculate_rbf_single(test_point[0], test_point[1], data_points)
    print(f"RBF result: {rbf_result:.1f}")
