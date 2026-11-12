import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# --- GEOSPATIAL FEATURE ENGINEERING HELPERS ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def distance_to_airport(lat, lon, airport_coords):
    return haversine_distance(lat, lon, airport_coords[0], airport_coords[1])

JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)
EWR_COORDS = (40.6895, -74.1745)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        p_lat = float(data['pickup_latitude'])
        p_lon = float(data['pickup_longitude'])
        d_lat = float(data['dropoff_latitude'])
        d_lon = float(data['dropoff_longitude'])
        passengers = int(data['passenger_count'])
        dt_str = data['pickup_datetime']
        
        # Parse Datetime
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
        
        # Calculate Features
        trip_distance_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)
        trip_distance_miles = trip_distance_km * 0.621371
        
        jfk_dist = min(distance_to_airport(p_lat, p_lon, JFK_COORDS), distance_to_airport(d_lat, d_lon, JFK_COORDS))
        lga_dist = min(distance_to_airport(p_lat, p_lon, LGA_COORDS), distance_to_airport(d_lat, d_lon, LGA_COORDS))
        
        # Enhanced NYC Taxi Fare Estimation Rules (Fallback / Baseline Model Integration)
        # Base Fare: $3.00 + $2.50 per mile + hour/peak surcharges
        base_fare = 3.00
        per_mile_rate = 3.30
        distance_charge = trip_distance_miles * per_mile_rate
        
        # Peak Hour Surcharge (4 PM - 8 PM weekdays)
        is_weekday = dt.weekday() < 5
        is_peak = is_weekday and (16 <= dt.hour <= 20)
        peak_surcharge = 2.50 if is_peak else 0.00
        
        # Night Surcharge (8 PM - 6 AM)
        is_night = (dt.hour >= 20 or dt.hour < 6)
        night_surcharge = 1.00 if is_night else 0.00
        
        # Flat rates for JFK
        if jfk_dist < 2.0:
            estimated_fare = 70.00 + peak_surcharge
        else:
            estimated_fare = base_fare + distance_charge + peak_surcharge + night_surcharge
            
        estimated_fare = max(2.50, round(estimated_fare, 2))
        
        return jsonify({
            'status': 'success',
            'predicted_fare': f"${estimated_fare:.2f}",
            'distance_miles': f"{trip_distance_miles:.2f} mi",
            'distance_km': f"{trip_distance_km:.2f} km",
            'pickup_time': dt.strftime("%A, %b %d %Y at %I:%M %p"),
            'details': {
                'peak_surcharge': f"${peak_surcharge:.2f}",
                'night_surcharge': f"${night_surcharge:.2f}",
                'passengers': passengers
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)