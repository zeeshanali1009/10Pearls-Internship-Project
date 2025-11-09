# feature_engineering.py
import os
import pandas as pd
from datetime import datetime
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURE_STORE_DIR = os.path.join(ROOT, 'feature_store')
TRAINING_TABLE = os.path.join(ROOT, 'training_data.parquet')

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(pm25):
    if pd.isna(pm25):
        return None
    for (clow, chigh, ilow, ihigh) in PM25_BREAKPOINTS:
        if clow <= pm25 <= chigh:
            aqi = (ihigh - ilow) / (chigh - clow) * (pm25 - clow) + ilow
            return round(aqi)
    return 501

def load_all_parquets(dirpath):
    files = glob.glob(os.path.join(dirpath, '*.parquet'))
    dfs = [pd.read_parquet(f) for f in files]
    if not dfs:
        raise RuntimeError('No parquet files in ' + dirpath)
    df = pd.concat(dfs).sort_index()
    return df

if __name__ == '__main__':
    df = load_all_parquets(FEATURE_STORE_DIR)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month

    df['pm25_lag1'] = df['pm2_5'].shift(1)
    df['pm25_rolling3'] = df['pm2_5'].rolling(window=3, min_periods=1).mean()
    df['pm25_change_rate'] = (df['pm2_5'] - df['pm25_lag1']) / (df['pm25_lag1'].replace(0, pd.NA))

    df['target_aqi'] = df['pm2_5'].apply(pm25_to_aqi)
    df = df[~df['target_aqi'].isna()]

    df.to_parquet(TRAINING_TABLE)
    print('Wrote training data to', TRAINING_TABLE)
