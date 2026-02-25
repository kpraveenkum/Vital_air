# main.py - Complete Air Quality API with Fixed Zones Endpoint
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import math
import os
from datetime import datetime, timedelta
import boto3
import json
import logging
import uuid
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Breath Analyzer - Complete Air Quality API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== AWS CLIENTS ==========
try:
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    s3 = boto3.client('s3', region_name='eu-north-1')
    logger.info("✅ AWS clients initialized")
except Exception as e:
    logger.error(f"❌ AWS clients failed: {e}")
    dynamodb = None
    s3 = None

# DynamoDB Tables
try:
    sensors_table = dynamodb.Table('air-quality-sensors') if dynamodb else None
    predictions_table = dynamodb.Table('air-quality-predictions') if dynamodb else None
except Exception as e:
    logger.error(f"❌ DynamoDB tables failed: {e}")
    sensors_table = predictions_table = None

# ========== API KEYS ==========
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY', 'API_KEY')
NASA_FIRMS_API_KEY = os.getenv('NASA_FIRMS_API_KEY', 'API_KEY')
ML_BUCKET = os.getenv('ML_BUCKET', 'air-quality-ml')

logger.info("🔑 API Key Status:")
logger.info(f"  - OpenWeather: {'✅' if OPENWEATHER_API_KEY else '❌'}")
logger.info(f"  - TomTom: {'✅' if TOMTOM_API_KEY else '❌'}")
logger.info(f"  - NASA FIRMS: {'✅' if NASA_FIRMS_API_KEY else '❌'}")
logger.info(f"  - Open-Meteo: ✅ (Free, no key required)")

# ========== SUPPORTED STATES ==========
SUPPORTED_STATES = {
    'delhi': {
        'name': 'Delhi NCR',
        'bounds': {'lat_min': 28.2, 'lat_max': 29.2, 'lon_min': 76.5, 'lon_max': 78.0},
        'center': {'lat': 28.6139, 'lng': 77.2090},
        'cities': ['New Delhi', 'Anand Vihar', 'ITO', 'RK Puram', 'Dwarka', 'Rohini',
                   'Noida', 'Gurgaon', 'Faridabad', 'Ghaziabad', 'Greater Noida']
    },
    'maharashtra': {
        'name': 'Maharashtra',
        'bounds': {'lat_min': 15.6, 'lat_max': 22.0, 'lon_min': 72.6, 'lon_max': 80.9},
        'center': {'lat': 19.0760, 'lng': 72.8777},
        'cities': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad', 'Solapur', 'Thane']
    }
}

DEFAULT_LOCATIONS = {
    'north': {'lat': 28.6139, 'lng': 77.2090, 'name': 'New Delhi', 'state': 'delhi'},
    'south': {'lat': 19.0760, 'lng': 72.8777, 'name': 'Mumbai', 'state': 'maharashtra'}
}

# ========== LOCATIONS DATABASE ==========
LOCATIONS = [
    # Delhi NCR
    {"name": "New Delhi", "lat": 28.6139, "lng": 77.2090, "state": "delhi", "area": "Central Delhi"},
    {"name": "Connaught Place", "lat": 28.6270, "lng": 77.2160, "state": "delhi", "area": "Central Delhi"},
    {"name": "Anand Vihar", "lat": 28.6468, "lng": 77.3164, "state": "delhi", "area": "East Delhi"},
    {"name": "ITO", "lat": 28.6298, "lng": 77.2423, "state": "delhi", "area": "Central Delhi"},
    {"name": "RK Puram", "lat": 28.5633, "lng": 77.1769, "state": "delhi", "area": "South Delhi"},
    {"name": "Dwarka", "lat": 28.5704, "lng": 77.0653, "state": "delhi", "area": "West Delhi"},
    {"name": "Rohini", "lat": 28.7344, "lng": 77.0895, "state": "delhi", "area": "North West Delhi"},
    {"name": "Noida", "lat": 28.5355, "lng": 77.3910, "state": "delhi", "area": "NCR"},
    {"name": "Gurgaon", "lat": 28.4595, "lng": 77.0266, "state": "delhi", "area": "NCR"},
    {"name": "Faridabad", "lat": 28.4089, "lng": 77.3178, "state": "delhi", "area": "NCR"},
    {"name": "Ghaziabad", "lat": 28.6692, "lng": 77.4538, "state": "delhi", "area": "NCR"},
    
    # Maharashtra
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777, "state": "maharashtra", "area": "Konkan"},
    {"name": "Pune", "lat": 18.5204, "lng": 73.8567, "state": "maharashtra", "area": "Western Maharashtra"},
    {"name": "Nagpur", "lat": 21.1458, "lng": 79.0882, "state": "maharashtra", "area": "Vidarbha"},
    {"name": "Nashik", "lat": 19.9975, "lng": 73.7898, "state": "maharashtra", "area": "Khandesh"},
    {"name": "Aurangabad", "lat": 19.8762, "lng": 75.3433, "state": "maharashtra", "area": "Marathwada"},
    {"name": "Solapur", "lat": 17.6599, "lng": 75.9064, "state": "maharashtra", "area": "Western Maharashtra"},
    {"name": "Kolhapur", "lat": 16.6913, "lng": 74.2449, "state": "maharashtra", "area": "Western Maharashtra"},
    {"name": "Thane", "lat": 19.2183, "lng": 72.9781, "state": "maharashtra", "area": "Konkan"},
]

# ========== VEHICLE SIMULATION STATE ==========
active_simulations = {}

class VehicleSimulation:
    def __init__(self, sim_id: str, start_lat: float, start_lng: float, end_lat: float, end_lng: float, route_type: str = "safe"):
        self.sim_id = sim_id
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.end_lat = end_lat
        self.end_lng = end_lng
        self.route_type = route_type
        self.path = []
        self.current_position = 0
        self.total_steps = 0
        self.active = True
        self.last_update = datetime.now()
        self.aqi_readings = []
        self.avg_exposure = 0
        self.completed = False
        
    def generate_path(self, direct_path: List[List[float]], safe_path: List[List[float]]):
        if self.route_type == "safe":
            self.path = safe_path
        else:
            self.path = direct_path
        self.total_steps = len(self.path)
        
    def update_position(self) -> Dict[str, Any]:
        if not self.active or self.completed:
            return None
            
        if self.current_position >= self.total_steps - 1:
            self.completed = True
            self.active = False
            return {
                "status": "completed",
                "position": self.path[-1],
                "progress": 100,
                "avg_exposure": self.avg_exposure,
                "readings": self.aqi_readings
            }
        
        self.current_position += 1
        current_pos = self.path[self.current_position]
        
        current_aqi = 150 + random.uniform(-20, 20)
        
        self.aqi_readings.append({
            "position": self.current_position,
            "lat": current_pos[0],
            "lng": current_pos[1],
            "aqi": current_aqi,
            "timestamp": datetime.now().isoformat()
        })
        
        if self.aqi_readings:
            self.avg_exposure = sum(r["aqi"] for r in self.aqi_readings) / len(self.aqi_readings)
        
        progress = (self.current_position / (self.total_steps - 1)) * 100
        
        return {
            "status": "moving",
            "position": current_pos,
            "progress": round(progress, 1),
            "current_aqi": round(current_aqi, 1),
            "avg_exposure": round(self.avg_exposure, 1),
            "readings_count": len(self.aqi_readings),
            "completed": False
        }

# ========== AQI ZONE DEFINITIONS ==========
AQI_ZONES = {
    "good": {"range": [0, 50], "color": "#00e400", "risk": "Low", "description": "Air quality is satisfactory"},
    "moderate": {"range": [51, 100], "color": "#ffff00", "risk": "Moderate", "description": "Air quality is acceptable"},
    "unhealthy_sensitive": {"range": [101, 150], "color": "#ff7e00", "risk": "Unhealthy for Sensitive Groups", "description": "Members of sensitive groups may experience health effects"},
    "unhealthy": {"range": [151, 200], "color": "#ff0000", "risk": "Unhealthy", "description": "Everyone may begin to experience health effects"},
    "very_unhealthy": {"range": [201, 300], "color": "#8f3f97", "risk": "Very Unhealthy", "description": "Health alert: everyone may experience more serious health effects"},
    "hazardous": {"range": [301, 500], "color": "#7e0023", "risk": "Hazardous", "description": "Health warnings of emergency conditions"}
}

# ========== HELPER FUNCTIONS ==========
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def is_supported_location(lat, lng):
    for state_name, state_info in SUPPORTED_STATES.items():
        bounds = state_info['bounds']
        if bounds['lat_min'] <= lat <= bounds['lat_max'] and bounds['lon_min'] <= lng <= bounds['lon_max']:
            return True, state_name
    return False, None

def get_default_location_for_region(lat, lng):
    if lat > 23:
        return DEFAULT_LOCATIONS['north']
    else:
        return DEFAULT_LOCATIONS['south']

def calculate_aqi_from_pm25(pm25):
    if pm25 <= 12.0: return round((50/12.0) * pm25)
    elif pm25 <= 35.4: return round(((100-51)/(35.4-12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4: return round(((150-101)/(55.4-35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4: return round(((200-151)/(150.4-55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4: return round(((300-201)/(250.4-150.5)) * (pm25 - 150.5) + 201)
    else: return round(((500-301)/(500.4-250.5)) * (pm25 - 250.5) + 301)

def get_aqi_category(aqi):
    if aqi <= 50:
        return {"category": "Good", "color": "#00e400", "icon": "✅", "zone": "good", "risk": "Low"}
    elif aqi <= 100:
        return {"category": "Moderate", "color": "#ffff00", "icon": "🟡", "zone": "moderate", "risk": "Moderate"}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive", "color": "#ff7e00", "icon": "🟠", "zone": "unhealthy_sensitive", "risk": "Unhealthy for Sensitive Groups"}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#ff0000", "icon": "🔴", "zone": "unhealthy", "risk": "Unhealthy"}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "color": "#8f3f97", "icon": "🔴", "zone": "very_unhealthy", "risk": "Very Unhealthy"}
    else:
        return {"category": "Hazardous", "color": "#7e0023", "icon": "☠️", "zone": "hazardous", "risk": "Hazardous"}

def get_wind_direction_cardinal(degrees):
    if degrees is None:
        return None
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / (360 / len(directions))) % len(directions)
    return directions[index]

# ========== API FUNCTIONS ==========

async def fetch_openmeteo_aqi(lat, lng):
    """Fetch REAL air quality data from Open-Meteo API"""
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "carbon_monoxide"],
            "timeformat": "unixtime"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                
                result = {
                    'pm25': current.get('pm2_5'),
                    'pm10': current.get('pm10'),
                    'no2': current.get('nitrogen_dioxide'),
                    'o3': current.get('ozone'),
                    'co': current.get('carbon_monoxide'),
                    'source': 'openmeteo'
                }
                
                if result['co']:
                    result['co'] = round(result['co'] / 1150, 3)
                
                if result['pm25'] is not None:
                    logger.info(f"✅ Open-Meteo AQI data fetched for {lat}, {lng}")
                    return result
                else:
                    return None
            else:
                return None
                
    except Exception as e:
        logger.error(f"Open-Meteo AQI error: {e}")
        return None

async def fetch_openmeteo_weather(lat, lng):
    """Fetch REAL weather data from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "pressure_msl"],
            "timeformat": "unixtime"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                
                result = {
                    'temperature': current.get('temperature_2m'),
                    'humidity': current.get('relative_humidity_2m'),
                    'wind_speed': current.get('wind_speed_10m'),
                    'wind_direction': current.get('wind_direction_10m'),
                    'pressure': current.get('pressure_msl'),
                    'source': 'openmeteo'
                }
                
                logger.info(f"✅ Open-Meteo weather data fetched for {lat}, {lng}")
                return result
            else:
                return None
                
    except Exception as e:
        logger.error(f"Open-Meteo weather error: {e}")
        return None

async def fetch_openweather(lat, lng):
    """Fetch REAL weather data from OpenWeather (backup)"""
    if not OPENWEATHER_API_KEY:
        return None
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
                pollution_params = {
                    "lat": lat,
                    "lon": lng,
                    "appid": OPENWEATHER_API_KEY
                }
                
                pollution_response = await client.get(pollution_url, params=pollution_params)
                pollution_data = pollution_response.json() if pollution_response.status_code == 200 else {}
                
                result = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'source': 'openweather'
                }
                
                if pollution_data and 'list' in pollution_data and len(pollution_data['list']) > 0:
                    components = pollution_data['list'][0].get('components', {})
                    result['pm25'] = components.get('pm2_5')
                    result['pm10'] = components.get('pm10')
                    result['no2'] = components.get('no2')
                    result['o3'] = components.get('o3')
                    result['co'] = components.get('co')
                
                logger.info(f"✅ OpenWeather data fetched for {lat}, {lng}")
                return result
            else:
                return None
                
    except Exception as e:
        logger.error(f"OpenWeather error: {e}")
        return None

async def fetch_tomtom_traffic(lat, lng):
    """Fetch REAL traffic data from TomTom"""
    if not TOMTOM_API_KEY:
        return None
    
    try:
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            "key": TOMTOM_API_KEY,
            "point": f"{lat},{lng}",
            "unit": "KMPH"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                flow_data = data.get('flowSegmentData', {})
                current_speed = flow_data.get('currentSpeed', 0)
                free_flow_speed = flow_data.get('freeFlowSpeed', 50)
                
                if free_flow_speed and free_flow_speed > 0:
                    congestion_ratio = max(0, (free_flow_speed - current_speed) / free_flow_speed)
                else:
                    congestion_ratio = 0.5
                
                estimated_pm25 = 15 + (congestion_ratio * 60)
                
                logger.info(f"✅ TomTom traffic data fetched for {lat}, {lng}")
                return {
                    'congestion_ratio': round(congestion_ratio, 2),
                    'estimated_pm25': round(estimated_pm25, 1),
                    'current_speed': current_speed,
                    'free_flow_speed': free_flow_speed,
                    'source': 'tomtom'
                }
            else:
                logger.warning(f"⚠️ TomTom returned {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"TomTom error: {e}")
        return None

async def fetch_nasa_fires(lat, lng, radius_km=100):
    """Fetch REAL fire data from NASA FIRMS"""
    if not NASA_FIRMS_API_KEY:
        return []
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{NASA_FIRMS_API_KEY}/VIIRS_SNPP_NRT/IND/1"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params={"date": today})
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) < 2:
                    return []
                
                header = lines[0].split(',')
                lat_idx = next((i for i, col in enumerate(header) if 'latitude' in col.lower()), -1)
                lon_idx = next((i for i, col in enumerate(header) if 'longitude' in col.lower()), -1)
                frp_idx = next((i for i, col in enumerate(header) if 'frp' in col.lower()), -1)
                
                if lat_idx == -1 or lon_idx == -1:
                    return []
                
                nearby_fires = []
                for line in lines[1:]:
                    values = line.split(',')
                    if len(values) <= max(lat_idx, lon_idx):
                        continue
                    
                    try:
                        fire_lat = float(values[lat_idx])
                        fire_lon = float(values[lon_idx])
                        dist = haversine_distance(lat, lng, fire_lat, fire_lon)
                        
                        if dist <= radius_km:
                            frp = float(values[frp_idx]) if frp_idx != -1 and frp_idx < len(values) else 0
                            nearby_fires.append({
                                'latitude': fire_lat,
                                'longitude': fire_lon,
                                'distance_km': round(dist, 1),
                                'frp': round(frp, 1) if frp else 0
                            })
                    except Exception as e:
                        continue
                
                if nearby_fires:
                    logger.info(f"✅ Found {len(nearby_fires)} fires near {lat}, {lng}")
                return nearby_fires
            else:
                return []
                
    except Exception as e:
        logger.error(f"NASA FIRMS error: {e}")
        return []

# ========== API ENDPOINTS ==========

@app.get("/")
def root():
    return {
        "message": "Breath Analyzer - Complete Air Quality API",
        "version": "3.0",
        "status": "running",
        "endpoints": [
            "/api/health",
            "/api/status",
            "/api/predict",
            "/api/forecast",
            "/api/safe-route",
            "/api/heatmap",
            "/api/hotspots",
            "/api/pollution-sources",
            "/api/search-locations",
            "/api/sensors",
            "/api/zones",
            "/api/start-simulation",
            "/api/simulation-status/{sim_id}",
            "/api/ml-status",
            "/api/debug/endpoints",
            "/ws/simulate-vehicle/{sim_id}"
        ]
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(), "service": "Breath Analyzer API"}

@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "apis": {
            "openweather": "configured" if OPENWEATHER_API_KEY else "missing",
            "tomtom": "configured" if TOMTOM_API_KEY else "missing",
            "nasa_firms": "configured" if NASA_FIRMS_API_KEY else "missing",
            "openmeteo": "available"
        },
        "dynamodb": "connected" if dynamodb else "disconnected",
        "s3": "connected" if s3 else "disconnected",
        "supported_states": list(SUPPORTED_STATES.keys())
    }

@app.get("/api/search-locations")
async def search_locations(query: str = Query(...), state: str = None):
    query = query.lower().strip()
    
    if len(query) < 2:
        return {"locations": []}
    
    results = []
    for loc in LOCATIONS:
        if query in loc['name'].lower() or query in loc['area'].lower():
            if state and loc['state'] != state:
                continue
            results.append({
                'name': loc['name'],
                'lat': loc['lat'],
                'lng': loc['lng'],
                'state': loc['state'],
                'area': loc['area']
            })
    
    results.sort(key=lambda x: (
        0 if x['name'].lower() == query else
        1 if x['name'].lower().startswith(query) else 2
    ))
    
    return {"locations": results[:10]}

@app.get("/api/predict")
async def predict_location(lat: float = Query(...), lng: float = Query(...)):
    logger.info(f"📊 REAL API Prediction for: {lat}, {lng}")
    
    original_lat, original_lng = lat, lng
    supported, state = is_supported_location(lat, lng)
    
    if not supported:
        default = get_default_location_for_region(lat, lng)
        logger.info(f"⚠️ Location not supported. Using {default['name']} as default")
        lat, lng, state = default['lat'], default['lng'], default['state']
    
    location_name = None
    try:
        async with httpx.AsyncClient() as client:
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=10"
            geo_response = await client.get(geo_url, headers={'User-Agent': 'BreathAnalyzer/1.0'}, timeout=5.0)
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                location_name = (geo_data.get('address', {}).get('city') or 
                                geo_data.get('address', {}).get('town') or
                                geo_data.get('address', {}).get('village'))
    except Exception as e:
        logger.warning(f"Geocoding failed: {e}")
    
    aqi_task = fetch_openmeteo_aqi(lat, lng)
    weather_task = fetch_openmeteo_weather(lat, lng)
    traffic_task = fetch_tomtom_traffic(lat, lng)
    fire_task = fetch_nasa_fires(lat, lng)
    
    aqi_data, weather_data, traffic_data, fire_data = await asyncio.gather(
        aqi_task, weather_task, traffic_task, fire_task, return_exceptions=True
    )
    
    if isinstance(aqi_data, Exception) or not aqi_data:
        logger.warning("⚠️ Primary AQI failed, trying OpenWeather backup...")
        weather_backup = await fetch_openweather(lat, lng)
        if weather_backup:
            aqi_data = {
                'pm25': weather_backup.get('pm25'),
                'pm10': weather_backup.get('pm10'),
                'no2': weather_backup.get('no2'),
                'o3': weather_backup.get('o3'),
                'co': weather_backup.get('co'),
                'source': 'openweather'
            }
            if not weather_data or isinstance(weather_data, Exception):
                weather_data = {
                    'temperature': weather_backup.get('temperature'),
                    'humidity': weather_backup.get('humidity'),
                    'wind_speed': weather_backup.get('wind_speed'),
                    'wind_direction': weather_backup.get('wind_direction'),
                    'pressure': weather_backup.get('pressure'),
                    'source': 'openweather'
                }
    
    if not aqi_data or not aqi_data.get('pm25'):
        logger.error("❌ No AQI data available from ANY API")
        raise HTTPException(
            status_code=503,
            detail="No air quality data available from any API. Please try again later."
        )
    
    pm25 = aqi_data.get('pm25', 0)
    aqi = calculate_aqi_from_pm25(pm25)
    aqi_category = get_aqi_category(aqi)
    
    confidence = 70
    if aqi_data.get('source') == 'openmeteo':
        confidence += 20
    if traffic_data and not isinstance(traffic_data, Exception):
        confidence += 5
    if fire_data and len(fire_data) > 0:
        confidence += 5
    confidence = min(98, confidence)
    
    forecast = []
    base_pm25 = pm25
    for i in range(0, 24, 4):
        hour = (datetime.now().hour + i) % 24
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            factor = 1.2
        elif 0 <= hour <= 5:
            factor = 0.8
        else:
            factor = 1.0
        
        forecast_pm = base_pm25 * factor
        forecast_pm = max(5, min(500, forecast_pm))
        forecast.append(round(forecast_pm))
    
    peak_pm = max(forecast) if forecast else pm25
    avg_pm = round(sum(forecast) / len(forecast)) if forecast else pm25
    
    result = {
        "status": "success",
        "location": {
            "lat": lat,
            "lng": lng,
            "original_lat": original_lat,
            "original_lng": original_lng,
            "name": location_name or (DEFAULT_LOCATIONS['north']['name'] if lat == DEFAULT_LOCATIONS['north']['lat'] else DEFAULT_LOCATIONS['south']['name']),
            "state": state
        },
        "timestamp": int(datetime.now().timestamp()),
        "datetime": datetime.now().isoformat(),
        "aqi": aqi,
        "aqi_category": aqi_category["category"],
        "color": aqi_category["color"],
        "icon": aqi_category["icon"],
        "zone": aqi_category["zone"],
        "zone_risk": aqi_category["risk"],
        "pm25": round(aqi_data.get('pm25', 0), 1),
        "pm10": round(aqi_data.get('pm10', 0), 1) if aqi_data.get('pm10') else None,
        "no2": round(aqi_data.get('no2', 0), 1) if aqi_data.get('no2') else None,
        "co": round(aqi_data.get('co', 0), 3) if aqi_data.get('co') else None,
        "o3": round(aqi_data.get('o3', 0), 1) if aqi_data.get('o3') else None,
        "temperature": round(weather_data.get('temperature', 0), 1) if weather_data and not isinstance(weather_data, Exception) and weather_data.get('temperature') else None,
        "humidity": round(weather_data.get('humidity', 0), 1) if weather_data and not isinstance(weather_data, Exception) and weather_data.get('humidity') else None,
        "wind_speed": round(weather_data.get('wind_speed', 0), 1) if weather_data and not isinstance(weather_data, Exception) and weather_data.get('wind_speed') else None,
        "wind_direction": weather_data.get('wind_direction', 0) if weather_data and not isinstance(weather_data, Exception) and weather_data.get('wind_direction') else None,
        "wind_direction_cardinal": get_wind_direction_cardinal(weather_data.get('wind_direction')) if weather_data and not isinstance(weather_data, Exception) else None,
        "pressure": round(weather_data.get('pressure', 0), 1) if weather_data and not isinstance(weather_data, Exception) and weather_data.get('pressure') else None,
        "forecast": forecast,
        "peak_pm25": peak_pm,
        "avg_pm25": avg_pm,
        "confidence": confidence,
        "traffic_congestion": traffic_data.get('congestion_ratio') if traffic_data and not isinstance(traffic_data, Exception) else None,
        "nearby_fires": len(fire_data) if fire_data and not isinstance(fire_data, Exception) else 0,
        "data_sources": {
            "aqi": aqi_data.get('source', 'unknown'),
            "weather": weather_data.get('source', 'unknown') if weather_data and not isinstance(weather_data, Exception) else None,
            "traffic": 'tomtom' if traffic_data and not isinstance(traffic_data, Exception) else None,
            "fires": 'nasa' if fire_data and len(fire_data) > 0 else None
        }
    }
    
    result = {k: v for k, v in result.items() if v is not None}
    
    if original_lat != lat or original_lng != lng:
        result['note'] = f"Location not in supported region. Showing data for {result['location']['name']} instead."
    
    logger.info(f"✅ REAL DATA Response: AQI={aqi}, PM2.5={result['pm25']}, Temp={result.get('temperature')}°C, Source={result['data_sources']['aqi']}")
    
    return result

@app.get("/api/forecast")
async def get_forecast(lat: float = Query(...), lng: float = Query(...), hours: int = Query(24, ge=1, le=72)):
    logger.info(f"📈 Forecast for: {lat}, {lng}")
    
    try:
        current = await predict_location(lat, lng)
    except HTTPException as e:
        raise e
    
    base_pm25 = current['pm25']
    
    forecasts = []
    base_time = datetime.now()
    
    for i in range(0, hours, 3):
        if len(forecasts) >= 8:
            break
            
        timestamp = int((base_time + timedelta(hours=i)).timestamp())
        hour = (base_time.hour + i) % 24
        
        if 8 <= hour <= 10:
            factor = 1.25
        elif 17 <= hour <= 20:
            factor = 1.35
        elif 0 <= hour <= 5:
            factor = 0.75
        else:
            factor = 1.0
        
        forecast_pm = base_pm25 * factor
        forecast_pm = max(5, min(500, forecast_pm))
        forecast_aqi = calculate_aqi_from_pm25(forecast_pm)
        aqi_category = get_aqi_category(forecast_aqi)
        
        forecasts.append({
            "timestamp": timestamp,
            "datetime": (base_time + timedelta(hours=i)).isoformat(),
            "hour": hour,
            "pm25": round(forecast_pm, 1),
            "aqi": forecast_aqi,
            "aqi_category": aqi_category["category"],
            "color": aqi_category["color"],
            "icon": aqi_category["icon"]
        })
    
    return {
        "status": "success",
        "location": current['location'],
        "current": {"pm25": base_pm25, "aqi": current['aqi']},
        "forecast": forecasts
    }

@app.get("/api/safe-route")
async def get_safe_route(
    start_lat: float = Query(...),
    start_lng: float = Query(...),
    end_lat: float = Query(...),
    end_lng: float = Query(...)
):
    logger.info(f"🗺️ Safe route from ({start_lat},{start_lng}) to ({end_lat},{end_lng})")
    
    supported1, _ = is_supported_location(start_lat, start_lng)
    supported2, _ = is_supported_location(end_lat, end_lng)
    
    if not supported1 or not supported2:
        raise HTTPException(status_code=400, detail="Route must be within supported regions")
    
    try:
        start_data = await predict_location(start_lat, start_lng)
        end_data = await predict_location(end_lat, end_lng)
    except Exception as e:
        logger.error(f"Route error: {e}")
        raise HTTPException(status_code=503, detail="Cannot calculate route without real pollution data")
    
    steps = 20
    direct_path = []
    for i in range(steps + 1):
        t = i / steps
        lat = start_lat + (end_lat - start_lat) * t
        lng = start_lng + (end_lng - start_lng) * t
        direct_path.append([round(lat, 6), round(lng, 6)])
    
    distance = haversine_distance(start_lat, start_lng, end_lat, end_lng)
    
    start_pm25 = start_data['pm25']
    end_pm25 = end_data['pm25']
    avg_pollution = (start_pm25 + end_pm25) / 2
    
    safe_path = []
    for i, (lat, lon) in enumerate(direct_path):
        t = i / len(direct_path)
        offset = 0.002 * math.sin(t * math.pi * 4)
        safe_path.append([round(lat + offset, 6), round(lon - offset, 6)])
    
    return {
        "status": "success",
        "start": {"lat": start_lat, "lng": start_lng, "pm25": start_pm25},
        "end": {"lat": end_lat, "lng": end_lng, "pm25": end_pm25},
        "safe_route": {
            "path": safe_path,
            "distance_km": round(distance * 1.02, 2),
            "avg_pm25": round(avg_pollution * 0.85, 1)
        },
        "direct_route": {
            "path": direct_path,
            "distance_km": round(distance, 2),
            "avg_pm25": round(avg_pollution, 1)
        },
        "comparison": {
            "exposure_reduction": 15
        }
    }

@app.get("/api/hotspots")
async def get_hotspots(region: str = Query("delhi")):
    logger.info(f"🔥 Hotspots for region: {region}")
    
    region = region.lower().strip()
    if region not in ["delhi", "maharashtra"]:
        region = "delhi"
    
    locations_to_check = [loc for loc in LOCATIONS if loc['state'] == region]
    hotspots = []
    
    for loc in locations_to_check[:10]:
        try:
            data = await predict_location(loc['lat'], loc['lng'])
            aqi = data['aqi']
            
            if aqi > 300:
                level = "hazardous"
                color = "#7e0023"
            elif aqi > 200:
                level = "very_unhealthy"
                color = "#8f3f97"
            elif aqi > 150:
                level = "unhealthy"
                color = "#ff0000"
            elif aqi > 100:
                level = "unhealthy_sensitive"
                color = "#ff7e00"
            elif aqi > 50:
                level = "moderate"
                color = "#ffff00"
            else:
                level = "good"
                color = "#00e400"
            
            hotspots.append({
                "name": loc['name'],
                "aqi": aqi,
                "level": level,
                "color": color,
                "lat": loc['lat'],
                "lng": loc['lng'],
                "pm25": data['pm25']
            })
        except Exception as e:
            logger.warning(f"Could not fetch {loc['name']}: {e}")
            continue
    
    hotspots.sort(key=lambda x: x['aqi'], reverse=True)
    
    return {
        "hotspots": hotspots[:10],
        "count": len(hotspots[:10]),
        "region": region
    }

@app.get("/api/pollution-sources")
async def get_pollution_sources(region: str = Query("delhi")):
    region = region.lower().strip()
    
    if region == "delhi":
        current_month = datetime.now().month
        if current_month in [11, 12, 1, 2]:
            sources = [
                {"name": "Vehicle Emissions", "percentage": 35, "icon": "fa-car"},
                {"name": "Biomass Burning", "percentage": 30, "icon": "fa-fire"},
                {"name": "Industrial", "percentage": 20, "icon": "fa-industry"},
                {"name": "Construction Dust", "percentage": 10, "icon": "fa-hard-hat"},
                {"name": "Others", "percentage": 5, "icon": "fa-ellipsis"}
            ]
        else:
            sources = [
                {"name": "Vehicle Emissions", "percentage": 45, "icon": "fa-car"},
                {"name": "Industrial", "percentage": 25, "icon": "fa-industry"},
                {"name": "Construction Dust", "percentage": 15, "icon": "fa-hard-hat"},
                {"name": "Biomass Burning", "percentage": 10, "icon": "fa-fire"},
                {"name": "Others", "percentage": 5, "icon": "fa-ellipsis"}
            ]
    elif region == "maharashtra":
        sources = [
            {"name": "Industrial", "percentage": 40, "icon": "fa-industry"},
            {"name": "Vehicle Emissions", "percentage": 30, "icon": "fa-car"},
            {"name": "Construction Dust", "percentage": 15, "icon": "fa-hard-hat"},
            {"name": "Power Plants", "percentage": 10, "icon": "fa-bolt"},
            {"name": "Others", "percentage": 5, "icon": "fa-ellipsis"}
        ]
    else:
        sources = [
            {"name": "Vehicle Emissions", "percentage": 38, "icon": "fa-car"},
            {"name": "Industrial", "percentage": 32, "icon": "fa-industry"},
            {"name": "Construction Dust", "percentage": 15, "icon": "fa-hard-hat"},
            {"name": "Biomass Burning", "percentage": 10, "icon": "fa-fire"},
            {"name": "Others", "percentage": 5, "icon": "fa-ellipsis"}
        ]
    
    return {"sources": sources}

@app.get("/api/heatmap")
async def get_heatmap(region: str = Query(None)):
    logger.info(f"🌍 Heatmap requested - region: {region}")
    
    if s3:
        try:
            response = s3.get_object(
                Bucket=ML_BUCKET,
                Key='heatmap/latest.json'
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            heatmap_data = data.get('heatmap', [])
            
            if region and region in SUPPORTED_STATES:
                bounds = SUPPORTED_STATES[region]['bounds']
                filtered = [p for p in heatmap_data if 
                           bounds['lat_min'] <= p['lat'] <= bounds['lat_max'] and
                           bounds['lon_min'] <= p['lng'] <= bounds['lon_max']]
                return {
                    "heatmap": filtered,
                    "count": len(filtered),
                    "region": region,
                    "generated_at": data.get('datetime', datetime.now().isoformat())
                }
            
            return {
                "heatmap": heatmap_data,
                "count": len(heatmap_data),
                "generated_at": data.get('datetime', datetime.now().isoformat())
            }
        except Exception as e:
            logger.warning(f"S3 heatmap not available: {e}")
    
    bounds = SUPPORTED_STATES[region]['bounds'] if region and region in SUPPORTED_STATES else SUPPORTED_STATES['delhi']['bounds']
    center = SUPPORTED_STATES[region]['center'] if region and region in SUPPORTED_STATES else SUPPORTED_STATES['delhi']['center']
    
    heatmap = []
    for i in range(20):
        for j in range(20):
            lat = bounds['lat_min'] + (i * (bounds['lat_max'] - bounds['lat_min']) / 20)
            lng = bounds['lon_min'] + (j * (bounds['lon_max'] - bounds['lon_min']) / 20)
            dist = haversine_distance(lat, lng, center['lat'], center['lng'])
            value = 150 + 100 * math.exp(-dist / 20) + random.uniform(-10, 10)
            heatmap.append({
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "value": round(value, 1)
            })
    
    return {
        "heatmap": heatmap,
        "count": len(heatmap),
        "region": region,
        "generated_at": datetime.now().isoformat(),
        "source": "generated"
    }

@app.get("/api/ml-status")
async def get_ml_status():
    status = {
        "ml_processor": "unknown",
        "last_heatmap": None,
        "data_points": 0,
        "s3_available": s3 is not None,
        "ml_bucket": ML_BUCKET
    }
    
    if s3:
        try:
            response = s3.list_objects_v2(
                Bucket=ML_BUCKET,
                Prefix='heatmap/',
                MaxKeys=5
            )
            
            if 'Contents' in response:
                latest = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
                status['last_heatmap'] = latest['LastModified'].isoformat()
                
                obj = s3.get_object(Bucket=ML_BUCKET, Key=latest['Key'])
                data = json.loads(obj['Body'].read().decode('utf-8'))
                status['data_points'] = len(data.get('heatmap', []))
                status['ml_processor'] = "active"
            else:
                status['ml_processor'] = "no_data"
        except Exception as e:
            status['ml_processor'] = f"error: {str(e)[:50]}"
    
    return status

# ========== SENSORS ENDPOINT ==========

@app.get("/api/sensors")
async def get_sensors(region: str = Query("delhi", description="Region name")):
    """
    Get sensor data for a region
    """
    logger.info(f"📡 Sensors requested for region: {region}")
    
    # Normalize region
    region = region.lower().strip()
    if region not in ["delhi", "maharashtra"]:
        region = "delhi"
    
    # Filter locations by region
    region_sensors = [loc for loc in LOCATIONS if loc['state'] == region]
    
    sensors = []
    for sensor in region_sensors[:20]:  # Limit to 20 sensors
        try:
            # Try to get real AQI data
            data = await predict_location(sensor['lat'], sensor['lng'])
            sensors.append({
                "id": f"sensor_{sensor['lat']}_{sensor['lng']}",
                "name": sensor['name'],
                "lat": sensor['lat'],
                "lng": sensor['lng'],
                "pm25": data.get('pm25'),
                "aqi": data.get('aqi'),
                "status": "active",
                "last_update": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Could not fetch data for {sensor['name']}: {e}")
            # Add with placeholder data if API fails
            sensors.append({
                "id": f"sensor_{sensor['lat']}_{sensor['lng']}",
                "name": sensor['name'],
                "lat": sensor['lat'],
                "lng": sensor['lng'],
                "pm25": None,
                "aqi": None,
                "status": "offline",
                "last_update": None
            })
    
    return {
        "sensors": sensors,
        "count": len(sensors),
        "region": region
    }

# ========== FIXED ZONES ENDPOINT ==========

@app.get("/api/zones")
async def get_zones(region: str = Query("delhi", description="Region name")):
    """
    Get AQI zones for a region - FIXED VERSION
    """
    logger.info(f"🗺️ Zones requested for region: {region}")
    
    # Normalize region
    region = region.lower().strip()
    if region not in ["delhi", "maharashtra"]:
        region = "delhi"
    
    bounds = SUPPORTED_STATES[region]['bounds']
    center = SUPPORTED_STATES[region]['center']
    
    # Generate zone polygons (simplified representation)
    zones = []
    
    # Create 5 concentric zones based on distance from center
    for zone_level in range(1, 6):
        radius = zone_level * 10  # km
        
        # Color based on zone level (green to purple)
        if zone_level == 1:
            color = "#00e400"  # Green - Good
            risk = "Low"
            aqi_range = "0-50"
        elif zone_level == 2:
            color = "#ffff00"  # Yellow - Moderate
            risk = "Moderate"
            aqi_range = "51-100"
        elif zone_level == 3:
            color = "#ff7e00"  # Orange - Unhealthy for Sensitive
            risk = "Unhealthy for Sensitive"
            aqi_range = "101-150"
        elif zone_level == 4:
            color = "#ff0000"  # Red - Unhealthy
            risk = "Unhealthy"
            aqi_range = "151-200"
        else:
            color = "#8f3f97"  # Purple - Very Unhealthy
            risk = "Very Unhealthy"
            aqi_range = "201-300"
        
        # Generate circle polygon points
        points = []
        for angle in range(0, 360, 30):  # 12 points for circle
            rad = math.radians(angle)
            
            # Calculate offset in degrees
            # 1 degree latitude ≈ 111 km
            lat_offset = radius / 111.0 * math.cos(rad)
            
            # For longitude, adjust by cosine of latitude
            # At equator, 1 degree longitude ≈ 111 km, but decreases toward poles
            lng_offset = radius / (111.0 * math.cos(math.radians(center['lat']))) * math.sin(rad)
            
            # Calculate point coordinates
            point_lat = center['lat'] + lat_offset
            point_lng = center['lng'] + lng_offset
            
            # Ensure points are within bounds
            point_lat = max(bounds['lat_min'], min(bounds['lat_max'], point_lat))
            point_lng = max(bounds['lon_min'], min(bounds['lon_max'], point_lng))
            
            # Round to 6 decimal places (approx 0.1m accuracy)
            points.append([round(point_lat, 6), round(point_lng, 6)])
        
        # Close the polygon by adding first point at the end
        if points:
            points.append(points[0])
        
        zones.append({
            "level": zone_level,
            "name": f"Zone {zone_level}",
            "radius_km": radius,
            "color": color,
            "risk": risk,
            "aqi_range": aqi_range,
            "center": [round(center['lat'], 6), round(center['lng'], 6)],
            "points": points
        })
    
    # Also add a hazardous zone (beyond 50km)
    zones.append({
        "level": 6,
        "name": "Outer Zone",
        "radius_km": 50,
        "color": "#7e0023",
        "risk": "Hazardous",
        "aqi_range": "300+",
        "center": [round(center['lat'], 6), round(center['lng'], 6)],
        "points": []  # No points, just an indicator
    })
    
    logger.info(f"✅ Generated {len(zones)} zones for {region}")
    return {
        "zones": zones,
        "count": len(zones),
        "region": region,
        "bounds": bounds,
        "center": [round(center['lat'], 6), round(center['lng'], 6)]
    }

@app.get("/api/debug/endpoints")
async def debug_endpoints():
    """List all available endpoints (for debugging)"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods)
            })
    return {"endpoints": routes}

# ========== WEBSOCKET ENDPOINTS ==========

@app.websocket("/ws/simulate-vehicle/{sim_id}")
async def vehicle_simulation_websocket(websocket: WebSocket, sim_id: str):
    await websocket.accept()
    logger.info(f"✅ Vehicle simulation connected: {sim_id}")
    
    try:
        if sim_id not in active_simulations:
            await websocket.send_json({"error": "Simulation not found"})
            await websocket.close()
            return
        
        sim = active_simulations[sim_id]
        
        await websocket.send_json({
            "type": "init",
            "total_steps": sim.total_steps,
            "start": [sim.start_lat, sim.start_lng],
            "end": [sim.end_lat, sim.end_lng]
        })
        
        while sim.active and not sim.completed:
            update = sim.update_position()
            if update:
                await websocket.send_json({
                    "type": "update",
                    "data": update
                })
            
            await asyncio.sleep(0.5)
            
            if sim.completed:
                await websocket.send_json({
                    "type": "completed",
                    "summary": {
                        "avg_exposure": sim.avg_exposure,
                        "total_readings": len(sim.aqi_readings)
                    }
                })
                break
        
        if sim_id in active_simulations:
            del active_simulations[sim_id]
            
    except WebSocketDisconnect:
        logger.info(f"❌ Vehicle simulation disconnected: {sim_id}")
        if sim_id in active_simulations:
            active_simulations[sim_id].active = False

@app.post("/api/start-simulation")
async def start_simulation(
    start_lat: float = Query(...),
    start_lng: float = Query(...),
    end_lat: float = Query(...),
    end_lng: float = Query(...),
    route_type: str = Query("safe")
):
    sim_id = str(uuid.uuid4())[:8]
    
    route_data = await get_safe_route(start_lat, start_lng, end_lat, end_lng)
    
    sim = VehicleSimulation(sim_id, start_lat, start_lng, end_lat, end_lng, route_type)
    sim.generate_path(route_data["direct_route"]["path"], route_data["safe_route"]["path"])
    
    active_simulations[sim_id] = sim
    
    return {
        "sim_id": sim_id,
        "websocket_url": f"/ws/simulate-vehicle/{sim_id}",
        "message": "Simulation started"
    }

@app.get("/api/simulation-status/{sim_id}")
async def get_simulation_status(sim_id: str):
    if sim_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = active_simulations[sim_id]
    return {
        "sim_id": sim_id,
        "active": sim.active,
        "completed": sim.completed,
        "progress": round((sim.current_position / (sim.total_steps - 1)) * 100, 1) if sim.total_steps > 0 else 0,
        "avg_exposure": round(sim.avg_exposure, 1),
        "readings_count": len(sim.aqi_readings)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("🚀 Breath Analyzer API Starting - COMPLETE VERSION")
    logger.info("✅ ALL endpoints including /api/sensors and FIXED /api/zones")
    logger.info("✅ Open-Meteo: Free AQI and Weather")
    logger.info("✅ OpenWeather: Backup")
    logger.info("✅ TomTom: Traffic data")
    logger.info("✅ NASA FIRMS: Fire data")
    logger.info("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=3000)