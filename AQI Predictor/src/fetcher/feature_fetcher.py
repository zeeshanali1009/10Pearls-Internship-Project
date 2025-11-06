# src/fetcher/feature_fetcher.py
"""
Feature fetcher for AQI project.

Modes:
 - Simulate data:    python feature_fetcher.py --simulate --days 7
 - Real OpenAQ fetch: python feature_fetcher.py --city "Karachi" --use_openaq
 - Real with OpenWeather: set OPENWEATHER_API_KEY env var and pass --use_openweather

Outputs:
 - data/raw/raw_<YYYY-MM-DD>.parquet   (appends hourly rows)
"""
import os
import argparse
from datetime import datetime, timedelta, timezone
import math
import json
import time

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

OPENAQ_ENDPOINT = "https://api.openaq.org/v2/measurements"
OPENWEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"

def simulate_rows_for_day(day_date, city="Karachi", freq="1H", seed=None):
    """
    Create simulated hourly pollutant + weather rows for a given date (UTC).
    Returns dataframe with 'timestamp_utc' and pollutant/weather cols.
    """
    if seed is not None:
        np.random.seed(seed + int(day_date.strftime("%Y%m%d")))
    idx = pd.date_range(start=pd.Timestamp(day_date, tz=timezone.utc),
                        periods=24, freq=freq)
    base_pm25 = 80 + 20 * np.sin(np.linspace(0, 3.14 * 2, 24))  # daily shape
    pm25 = base_pm25 + np.random.normal(scale=8.0, size=24)
    pm10 = pm25 * (1.5 + np.random.normal(scale=0.1, size=24))
    o3 = np.clip(30 + 10 * np.cos(np.linspace(0, 3.14 * 2, 24)) + np.random.normal(scale=5, size=24), 1, None)
    no2 = np.clip(20 + np.random.normal(scale=4.0, size=24), 0.1, None)
    so2 = np.clip(5 + np.random.normal(scale=2.0, size=24), 0.0, None)
    co = np.clip(0.6 + np.random.normal(scale=0.2, size=24), 0.01, None)
    temp = 25 + 3 * np.sin(np.linspace(0, 3.14*2, 24)) + np.random.normal(scale=1.5, size=24)
    humidity = np.clip(55 + 10 * np.cos(np.linspace(0, 3.14*2, 24)) + np.random.normal(scale=5.0, size=24), 10, 100)
    wind_speed = np.clip(3 + np.abs(np.random.normal(scale=1.2, size=24)), 0.0, None)

    rows = []
    for i, ts in enumerate(idx):
        rows.append({
            "timestamp_utc": ts.isoformat(),
            "city": city,
            "pm2_5": float(round(max(0.0, pm25[i]), 3)),
            "pm10": float(round(max(0.0, pm10[i]), 3)),
            "o3": float(round(o3[i], 3)),
            "no2": float(round(no2[i], 3)),
            "so2": float(round(so2[i], 3)),
            "co": float(round(co[i], 3)),
            "temp_c": float(round(temp[i], 2)),
            "humidity": float(round(humidity[i], 2)),
            "wind_speed": float(round(wind_speed[i], 2)),
        })
    return pd.DataFrame(rows)

def fetch_openaq_for_city_hour(city, dt_utc):
    """
    Query OpenAQ for measurements for a city at a timestamp (minute resolution).
    OpenAQ allows filtering by datetime (ISO) and returns many records - we will aggregate by parameter.
    """
    # OpenAQ wants local datetime range; we'll query +/- 30 minutes window
    dt_from = (dt_utc - timedelta(minutes=30)).isoformat()
    dt_to = (dt_utc + timedelta(minutes=30)).isoformat()
    params = {
        "city": city,
        "date_from": dt_from,
        "date_to": dt_to,
        "limit": 100,
        "page": 1,
        "offset": 0,
        "sort": "desc"
    }
    try:
        r = requests.get(OPENAQ_ENDPOINT, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        # aggregate latest values by parameter
        agg = {}
        for item in results:
            param = item.get("parameter")
            value = item.get("value")
            if param and value is not None:
                # keep latest (by date)
                if param not in agg or item.get("date", {}).get("utc", "") > agg[param][1]:
                    agg[param] = (value, item.get("date", {}).get("utc", ""))
        # map to our columns
        row = {
            "pm2_5": agg.get("pm25", (None, None))[0],
            "pm10": agg.get("pm10", (None, None))[0],
            "o3": agg.get("o3", (None, None))[0],
            "no2": agg.get("no2", (None, None))[0],
            "so2": agg.get("so2", (None, None))[0],
            "co": agg.get("co", (None, None))[0],
        }
        return row
    except Exception as e:
        # On any failure return empty row (NaNs)
        return {"pm2_5": None, "pm10": None, "o3": None, "no2": None, "so2": None, "co": None}

def fetch_openweather_for_city(city, api_key):
    """
    Simple current weather fetch for a city using OpenWeatherMap.
    Returns a dict with temp_c, humidity, wind_speed.
    """
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(OPENWEATHER_ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    return {
        "temp_c": main.get("temp"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed")
    }

def append_parquet_for_date(df_rows, day_date):
    """Append or create a parquet file for the day (UTC date)."""
    fname = os.path.join(RAW_DIR, f"raw_{day_date.strftime('%Y-%m-%d')}.parquet")
    if os.path.exists(fname):
        old = pd.read_parquet(fname)
        out = pd.concat([old, df_rows], ignore_index=True)
    else:
        out = df_rows
    # sort by timestamp
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"])
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out.to_parquet(fname, index=False)
    print(f"Saved {len(df_rows)} rows to {fname}")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true", help="Run in simulate mode (no external API calls)")
    p.add_argument("--days", type=int, default=3, help="Number of recent days to fetch/simulate (default 3)")
    p.add_argument("--city", type=str, default="Karachi", help="City name to fetch for (OpenAQ/OpenWeather)")
    p.add_argument("--use_openaq", action="store_true", help="Enable OpenAQ fetch (no API key required)")
    p.add_argument("--use_openweather", action="store_true", help="Enable OpenWeather fetch (api key required via OPENWEATHER_API_KEY env var)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between hourly fetches (useful for real fetching)")
    args = p.parse_args(argv)

    city = args.city
    end_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = end_utc - timedelta(days=args.days-1)
    dates = []
    # We'll iterate by day to create per-day parquet files
    for d in range(args.days):
        day_date = (end_utc - timedelta(days=d)).date()
        dates.append(day_date)
    dates = sorted(set(dates))

    if args.simulate:
        # For each day generate 24 rows (UTC) and save
        for day_date in dates:
            df = simulate_rows_for_day(day_date, city=city, seed=42)
            # timestamp is tz-aware ISO; convert to str
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_convert("UTC").astype(str)
            append_parquet_for_date(df, day_date)
        return

    # Real fetch mode
    # For each date we will iterate hours 0..23 (UTC)
    openweather_key = os.environ.get("OPENWEATHER_API_KEY")
    for day_date in dates:
        rows = []
        for hour in range(24):
            ts = datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour)
            # default row
            row = {
                "timestamp_utc": ts.isoformat(),
                "city": city,
                "pm2_5": None,
                "pm10": None,
                "o3": None,
                "no2": None,
                "so2": None,
                "co": None,
                "temp_c": None,
                "humidity": None,
                "wind_speed": None,
            }
            # try OpenAQ (may return some of the pollutants)
            if args.use_openaq:
                try:
                    aq = fetch_openaq_for_city_hour(city, ts)
                    row.update(aq)
                except Exception:
                    pass
            # try OpenWeather if requested and key present
            if args.use_openweather and openweather_key:
                try:
                    w = fetch_openweather_for_city(city, openweather_key)
                    row.update(w)
                except Exception:
                    pass
            rows.append(row)
            if args.sleep:
                time.sleep(args.sleep)
        df = pd.DataFrame(rows)
        append_parquet_for_date(df, day_date)
    print("Completed real fetch run.")

if __name__ == "__main__":
    main()
