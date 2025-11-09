# app_streamlit.py
import streamlit as st
import joblib
import os
import pandas as pd
import json
from datetime import datetime
import requests

MODEL_DIR = 'models'
CONFIG_PATH = 'config.json'

st.title('Pearls AQI Predictor — 72 hour forecast')

models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
if not models:
    st.error('No model found. Run scripts/train_model.py first.')
    st.stop()
models.sort()
model_info = joblib.load(os.path.join(MODEL_DIR, models[-1]))
model = model_info['model']
FEATURE_COLS = model_info['features']

st.write('Loaded model:', models[-1])

cfg = {}
if os.path.exists(CONFIG_PATH):
    cfg = json.load(open(CONFIG_PATH))
API_KEY = cfg.get('OPENWEATHER_API_KEY') or st.text_input('OpenWeather API Key', type='password')

if not API_KEY:
    st.info('Enter your OpenWeather API key to fetch latest forecast data')
    st.stop()

LAT = cfg.get('LAT')
LON = cfg.get('LON')
if not LAT or not LON:
    st.info('No LAT/LON in config.json — using city geocode')
    city = st.text_input('City (e.g. Lahore,PK)', value=cfg.get('CITY') or 'Lahore,PK')
    if st.button('Geocode'):
        url = 'http://api.openweathermap.org/geo/1.0/direct'
        r = requests.get(url, params={'q': city, 'limit': 1, 'appid': API_KEY})
        r.raise_for_status()
        data = r.json()
        if data:
            LAT, LON = data[0]['lat'], data[0]['lon']
            st.success(f'Geocoded to {LAT},{LON}')

if st.button('Fetch & Predict next 72 hours'):
    url = 'http://api.openweathermap.org/data/2.5/air_pollution/forecast'
    r = requests.get(url, params={'lat': LAT, 'lon': LON, 'appid': API_KEY})
    r.raise_for_status()
    resp = r.json()
    recs = []
    for item in resp.get('list', [])[:72]:
        ts = datetime.utcfromtimestamp(item['dt'])
        rr = {'ts': ts}
        comps = item.get('components', {})
        for k, v in comps.items():
            rr[k] = v
        recs.append(rr)
    df = pd.DataFrame(recs).set_index('ts')

    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['pm25_lag1'] = df['pm2_5'].shift(1).fillna(df['pm2_5'])
    df['pm25_rolling3'] = df['pm2_5'].rolling(3, min_periods=1).mean()
    df['pm25_change_rate'] = (df['pm2_5'] - df['pm25_lag1']) / df['pm25_lag1'].replace(0, 1)

    X = df[FEATURE_COLS].fillna(0)
    preds = model.predict(X)
    df['pred_aqi'] = preds

    st.line_chart(df['pred_aqi'])
    st.table(df[['pm2_5','pred_aqi']].head(24))

    hazardous = df[df['pred_aqi'] >= 151]
    if not hazardous.empty:
        st.warning(f'{len(hazardous)} hourly predictions in the next 72h are Hazardous (AQI>=151).')
