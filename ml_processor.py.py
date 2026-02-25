import json
import boto3
import os
import math
import time
import random
from datetime import datetime
import numpy as np

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Tables
sensors_table = dynamodb.Table(os.environ.get('SENSORS_TABLE', 'air-quality-sensors'))
predictions_table = dynamodb.Table(os.environ.get('PREDICTIONS_TABLE', 'air-quality-predictions'))

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_mock_data():
    """Generate mock sensor data for testing"""
    print("📊 Generating mock sensor data for testing...")
    
    # Delhi NCR locations
    mock_sensors = [
        {"lat": 28.6139, "lon": 77.2090, "value": 180, "name": "New Delhi"},      # High pollution
        {"lat": 28.6468, "lon": 77.3164, "value": 220, "name": "Anand Vihar"},    # Very high
        {"lat": 28.6298, "lon": 77.2423, "value": 195, "name": "ITO"},            # High
        {"lat": 28.5633, "lon": 77.1769, "value": 145, "name": "RK Puram"},       # Moderate
        {"lat": 28.5704, "lon": 77.0653, "value": 135, "name": "Dwarka"},         # Moderate
        {"lat": 28.5355, "lon": 77.3910, "value": 165, "name": "Noida"},          # High
        {"lat": 28.4595, "lon": 77.0266, "value": 155, "name": "Gurgaon"},        # High
        
        # Maharashtra locations
        {"lat": 19.0760, "lon": 72.8777, "value": 120, "name": "Mumbai"},         # Moderate
        {"lat": 18.5204, "lon": 73.8567, "value": 95, "name": "Pune"},            # Moderate
        {"lat": 21.1458, "lon": 79.0882, "value": 85, "name": "Nagpur"},          # Good
    ]
    
    data_points = []
    for sensor in mock_sensors:
        data_points.append({
            'lat': sensor['lat'],
            'lon': sensor['lon'],
            'value': sensor['value'],
            'weight': 1.0,
            'source': 'mock',
            'name': sensor['name']
        })
    
    print(f"✅ Generated {len(data_points)} mock data points")
    return data_points

def calculate_idw(target_lat, target_lon, data_points, power=2):
    """Inverse Distance Weighting interpolation"""
    if not data_points:
        return 100
    
    weighted_sum = 0
    total_weight = 0
    
    for point in data_points:
        dist = haversine_distance(target_lat, target_lon, point['lat'], point['lon'])
        
        if dist < 0.1:
            return point['value']
        
        weight = 1.0 / (dist ** power) * point.get('weight', 1.0)
        weighted_sum += weight * point['value']
        total_weight += weight
    
    if total_weight == 0:
        return 100
    
    return weighted_sum / total_weight

def generate_grid_points(bounds, density=50):
    """Generate grid points within bounds"""
    grid_points = []
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / density
    lon_step = (bounds['lon_max'] - bounds['lon_min']) / density
    
    for i in range(density):
        lat = bounds['lat_min'] + i * lat_step
        for j in range(density):
            lon = bounds['lon_min'] + j * lon_step
            grid_points.append((lat, lon))
    
    return grid_points

def lambda_handler(event, context):
    print("🧠 ML Processor started")
    start_time = time.time()
    timestamp = int(start_time)
    
    # Try to get real data from DynamoDB
    data_points = []
    try:
        response = sensors_table.scan(Limit=50)
        items = response.get('Items', [])
        
        for item in items:
            try:
                data_points.append({
                    'lat': float(item['latitude']),
                    'lon': float(item['longitude']),
                    'value': float(item.get('pm25', 100)),
                    'weight': 0.9,
                    'source': 'dynamodb'
                })
            except:
                continue
        
        print(f"📊 Found {len(data_points)} real sensor records")
    except Exception as e:
        print(f"⚠️ DynamoDB scan failed: {e}")
    
    # If no real data, use mock data
    if not data_points:
        print("⚠️ No real sensor data available, using mock data")
        data_points = generate_mock_data()
    
    # Define state bounds
    state_bounds = {
        'delhi': {
            'lat_min': 28.4, 'lat_max': 28.9,
            'lon_min': 76.8, 'lon_max': 77.3
        },
        'maharashtra': {
            'lat_min': 15.6, 'lat_max': 22.0,
            'lon_min': 72.6, 'lon_max': 80.9
        }
    }
    
    all_predictions = []
    
    # Generate predictions for each state
    for state, bounds in state_bounds.items():
        print(f"\n📍 Processing {state}...")
        grid_points = generate_grid_points(bounds, density=30)  # 30x30 = 900 points
        
        predictions = []
        for lat, lon in grid_points:
            value = calculate_idw(lat, lon, data_points)
            if value:
                predictions.append({
                    'lat': round(lat, 4),
                    'lng': round(lon, 4),
                    'value': round(value, 1)
                })
        
        print(f"   Generated {len(predictions)} predictions for {state}")
        all_predictions.extend(predictions)
    
    # Save to S3
    try:
        heatmap_data = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'total_predictions': len(all_predictions),
            'data_points_used': len(data_points),
            'heatmap': all_predictions[:1000],  # Limit for now
            'metadata': {
                'processing_time': round(time.time() - start_time, 2),
                'data_source': 'mock' if data_points[0].get('source') == 'mock' else 'real'
            }
        }
        
        # Save to S3
        s3.put_object(
            Bucket=os.environ.get('ML_BUCKET', 'air-quality-ml'),
            Key='heatmap/latest.json',
            Body=json.dumps(heatmap_data),
            ContentType='application/json'
        )
        print(f"✅ Saved heatmap to S3: {len(all_predictions)} predictions")
    except Exception as e:
        print(f"⚠️ S3 save failed: {e}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'ML processor completed',
            'predictions': len(all_predictions),
            'data_points': len(data_points),
            'processing_time': round(time.time() - start_time, 2),
            'data_source': 'mock' if data_points[0].get('source') == 'mock' else 'real'
        })
    }
