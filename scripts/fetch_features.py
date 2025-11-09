# fetch_features.py
# Fetches historical hourly air pollution + weather from OpenWeather's Air Pollution API
# Usage: python scripts/fetch_features.py --start 2025-01-01 --end 2025-11-01 --city "Lahore,PK"

import os
import argparse
import json
from datetime import datetime, timedelta
import time
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__) if '__file__' in globals() else '.'
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CONFIG_PATH = os.path.join(ROOT, 'config.json')
FEATURE_STORE_DIR = os.path.join(ROOT, 'feature_store')
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

# Load config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
else:
    cfg = {
        'OPENWEATHER_API_KEY': os.environ.get('OPENWEATHER_API_KEY'),
        'LAT': None,
        'LON': None,
        'CITY': None
    }

API_KEY = cfg.get('OPENWEATHER_API_KEY')
if not API_KEY:
    raise ValueError('Put your OpenWeather API key either in config.json or env var OPENWEATHER_API_KEY')

LAT = cfg.get('LAT')
LON = cfg.get('LON')

# Helper: geocode a city to lat/lon using OpenWeather
def geocode_city(city_name):
    url = 'http://api.openweathermap.org/geo/1.0/direct'
    params = {'q': city_name, 'limit': 1, 'appid': API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError('Unable to geocode city: ' + city_name)
    return data[0]['lat'], data[0]['lon']

HISTORY_ENDPOINT = 'http://api.openweathermap.org/data/2.5/air_pollution/history'
CURRENT_ENDPOINT = 'http://api.openweathermap.org/data/2.5/air_pollution'
FORECAST_ENDPOINT = 'http://api.openweathermap.org/data/2.5/air_pollution/forecast'

def fetch_hourly_pollution(lat, lon, start_unix, end_unix):
    params = {'lat': lat, 'lon': lon, 'start': int(start_unix), 'end': int(end_unix), 'appid': API_KEY}
    r = requests.get(HISTORY_ENDPOINT, params=params)
    r.raise_for_status()
    return r.json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--city', required=False)
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    if args.city:
        LAT, LON = geocode_city(args.city)
    if LAT is None or LON is None:
        raise ValueError('Provide LAT/LON in config.json or pass --city')

    all_records = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=6), end_dt)
        print(f'Fetching {chunk_start} to {chunk_end}')
        try:
            resp = fetch_hourly_pollution(LAT, LON, time.mktime(chunk_start.timetuple()), time.mktime(chunk_end.timetuple()))
        except Exception as e:
            print('Error fetching chunk:', e)
            break
        for item in resp.get('list', []):
            rec = {
                'ts': datetime.utcfromtimestamp(item['dt']),
                'aqi_owm': item.get('main', {}).get('aqi'),
            }
            comps = item.get('components', {})
            for k, v in comps.items():
                rec[k] = v
            all_records.append(rec)
        chunk_start = chunk_end + timedelta(seconds=1)
        time.sleep(1)

    if not all_records:
        print('No records fetched — check API limits and date range')
    else:
        df = pd.DataFrame(all_records).set_index('ts').sort_index()
        out_path = os.path.join(FEATURE_STORE_DIR, f'pollution_{start_dt.date()}_{end_dt.date()}.parquet')
        df.to_parquet(out_path)
        print('Saved', out_path)
