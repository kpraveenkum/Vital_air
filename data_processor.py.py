"""
data_processor.py - Fetches real AQI data from multiple APIs and stores in DynamoDB
Run this Lambda every hour to collect fresh data
"""

import json
import boto3
import requests
import time
import os
from datetime import datetime
from decimal import Decimal

# AWS Clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Tables
sensors_table = dynamodb.Table(os.environ.get('SENSORS_TABLE', 'air-quality-sensors'))
historical_table = dynamodb.Table(os.environ.get('HISTORICAL_TABLE', 'air-quality-historical'))

# API Keys (set these in Lambda environment variables)
OPENAQ_API_KEY = os.environ.get('OPENAQ_API_KEY', '')
WAQI_TOKEN = os.environ.get('WAQI_TOKEN', '')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')

# Supported locations with coordinates
LOCATIONS = {
    'delhi': [
        {'name': 'New Delhi', 'lat': 28.6139, 'lon': 77.2090},
        {'name': 'Anand Vihar', 'lat': 28.6468, 'lon': 77.3164},
        {'name': 'ITO', 'lat': 28.6298, 'lon': 77.2423},
        {'name': 'RK Puram', 'lat': 28.5633, 'lon': 77.1769},
        {'name': 'Dwarka', 'lat': 28.5704, 'lon': 77.0653},
        {'name': 'Rohini', 'lat': 28.7344, 'lon': 77.0895},
        {'name': 'Noida', 'lat': 28.5355, 'lon': 77.3910},
        {'name': 'Gurgaon', 'lat': 28.4595, 'lon': 77.0266},
    ],
    'maharashtra': [
        {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
        {'name': 'Pune', 'lat': 18.5204, 'lon': 73.8567},
        {'name': 'Nagpur', 'lat': 21.1458, 'lon': 79.0882},
        {'name': 'Nashik', 'lat': 19.9975, 'lon': 73.7898},
        {'name': 'Aurangabad', 'lat': 19.8762, 'lon': 75.3433},
    ]
}

def fetch_openaq_data(lat, lon, location_name):
    """Fetch PM2.5 data from OpenAQ API"""
    try:
        url = "https://api.openaq.org/v2/latest"
        params = {
            'coordinates': f"{lat},{lon}",
            'radius': 10000,  # 10km radius
            'limit': 10
        }
        headers = {'X-API-Key': OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        pm25_values = []
        for result in data.get('results', []):
            for measurement in result.get('measurements', []):
                if measurement.get('parameter') == 'pm25':
                    pm25_values.append(measurement.get('value', 0))
        
        if pm25_values:
            avg_pm25 = sum(pm25_values) / len(pm25_values)
            return {
                'pm25': round(avg_pm25, 1),
                'source': 'openaq',
                'station_count': len(pm25_values)
            }
    except Exception as e:
        print(f"OpenAQ error for {location_name}: {e}")
    return None

def fetch_waqi_data(lat, lon, location_name):
    """Fetch AQI data from WAQI API (backup)"""
    if not WAQI_TOKEN:
        return None
    
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
        params = {'token': WAQI_TOKEN}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') == 'ok':
            iaqi = data.get('data', {}).get('iaqi', {})
            pm25 = iaqi.get('pm25', {}).get('v')
            if pm25:
                return {
                    'pm25': round(float(pm25), 1),
                    'source': 'waqi',
                    'station_count': 1
                }
    except Exception as e:
        print(f"WAQI error for {location_name}: {e}")
    return None

def fetch_openweather_data(lat, lon, location_name):
    """Fetch weather data (helps with confidence scoring)"""
    if not OPENWEATHER_API_KEY:
        return None
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'weather': data['weather'][0]['description']
        }
    except Exception as e:
        print(f"OpenWeather error for {location_name}: {e}")
    return None

def save_to_dynamodb(sensor_data):
    """Save sensor reading to DynamoDB"""
    try:
        item = {
            'sensor_id': f"{sensor_data['location']}_{int(time.time())}",
            'timestamp': int(time.time()),
            'datetime': datetime.now().isoformat(),
            'latitude': Decimal(str(sensor_data['lat'])),
            'longitude': Decimal(str(sensor_data['lon'])),
            'pm25': Decimal(str(sensor_data['pm25'])),
            'source': sensor_data['source'],
            'location_name': sensor_data['location'],
            'state': sensor_data['state'],
            'station_count': sensor_data.get('station_count', 1),
            'confidence': sensor_data.get('confidence', 80)
        }
        
        # Add weather data if available
        if 'weather' in sensor_data:
            item['temperature'] = Decimal(str(sensor_data['weather']['temperature']))
            item['humidity'] = Decimal(str(sensor_data['weather']['humidity']))
            item['wind_speed'] = Decimal(str(sensor_data['weather']['wind_speed']))
        
        sensors_table.put_item(Item=item)
        
        # Also save to historical table (for long-term trends)
        historical_table.put_item(Item={
            **item,
            'sensor_id': f"hist_{item['sensor_id']}"
        })
        
        return True
    except Exception as e:
        print(f"DynamoDB error: {e}")
        return False

def lambda_handler(event, context):
    """Main data processor handler"""
    print("🌍 DATA PROCESSOR: Fetching real AQI data from APIs")
    
    timestamp = int(time.time())
    saved_count = 0
    all_sensors = []
    
    # Process each state and location
    for state, locations in LOCATIONS.items():
        print(f"\n📍 Processing {state.upper()}...")
        
        for location in locations:
            print(f"   Fetching data for {location['name']}...")
            
            # Try OpenAQ first (primary source)
            sensor_data = fetch_openaq_data(
                location['lat'], 
                location['lon'], 
                location['name']
            )
            
            # If OpenAQ fails, try WAQI (backup)
            if not sensor_data:
                print(f"   ⚠️ OpenAQ failed, trying WAQI...")
                sensor_data = fetch_waqi_data(
                    location['lat'], 
                    location['lon'], 
                    location['name']
                )
            
            if sensor_data:
                # Fetch weather data for context
                weather = fetch_openweather_data(
                    location['lat'], 
                    location['lon'], 
                    location['name']
                )
                
                # Calculate confidence score
                confidence = 70  # Base confidence
                if sensor_data['source'] == 'openaq':
                    confidence += 20
                if sensor_data.get('station_count', 0) > 3:
                    confidence += 10
                
                # Prepare full sensor record
                full_record = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'location': location['name'],
                    'state': state,
                    'pm25': sensor_data['pm25'],
                    'source': sensor_data['source'],
                    'station_count': sensor_data.get('station_count', 1),
                    'confidence': min(98, confidence),
                    'timestamp': timestamp
                }
                
                if weather:
                    full_record['weather'] = weather
                
                # Save to DynamoDB
                if save_to_dynamodb(full_record):
                    saved_count += 1
                    all_sensors.append(full_record)
                    print(f"   ✅ Saved: {location['name']} - PM2.5: {sensor_data['pm25']} µg/m³")
                else:
                    print(f"   ❌ Failed to save {location['name']}")
            else:
                print(f"   ❌ No data for {location['name']}")
    
    # Also fetch from historical data (last 24 hours)
    print(f"\n📊 Fetching historical data (last 24 hours)...")
    historical_data = fetch_historical_data()
    
    # Save all sensor data to S3 for ML processor
    try:
        s3.put_object(
            Bucket=os.environ.get('ML_BUCKET', 'air-quality-ml'),
            Key=f"raw/sensors_{timestamp}.json",
            Body=json.dumps({
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'sensors': all_sensors,
                'historical_count': len(historical_data)
            }, default=str),
            ContentType='application/json'
        )
        
        # Also save as latest.json for easy access
        s3.put_object(
            Bucket=os.environ.get('ML_BUCKET', 'air-quality-ml'),
            Key="raw/latest.json",
            Body=json.dumps({
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'sensors': all_sensors
            }, default=str),
            ContentType='application/json'
        )
        
        print(f"\n✅ Saved {saved_count} sensors to DynamoDB and S3")
    except Exception as e:
        print(f"⚠️ S3 save error: {e}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Data processor completed',
            'sensors_saved': saved_count,
            'timestamp': timestamp,
            'states': list(LOCATIONS.keys())
        })
    }

def fetch_historical_data():
    """Fetch last 24 hours of historical data"""
    try:
        # Get last 24 hours timestamp
        cutoff = int(time.time()) - (24 * 3600)
        
        response = historical_table.scan(
            FilterExpression='#ts > :cutoff',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={':cutoff': cutoff}
        )
        
        return response.get('Items', [])
    except Exception as e:
        print(f"Historical fetch error: {e}")
        return []

# For local testing
if __name__ == "__main__":
    print("🧪 Testing Data Processor locally...")
    result = lambda_handler({'test': True}, {})
    print(json.dumps(result, indent=2))
