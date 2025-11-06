# src/features/feature_engineer.py
"""
Feature engineering script.

Reads data/raw/*.parquet (expects 'timestamp_utc', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co',
'temp_c', 'humidity', 'wind_speed', 'city'), resamples to 1H, interpolates missing, computes:
 - time features (hour, dayofweek, month)
 - rolling means for pm2_5 (3h, 6h, 24h)
 - diffs and pct changes
 - lag features (1h, 3h)
 - lead targets for prediction (1h,3h,6h,24h,72h): pm2_5_lead_<H>h

Outputs to data/features/features_<YYYY-MM-DD>.parquet
"""
import os
from datetime import datetime, timezone
import argparse

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
FEATURE_DIR = os.path.join(ROOT, "data", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

def load_all_raw():
    files = sorted([os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError(f"No raw parquet files in {RAW_DIR}. Run fetcher first.")
    dfs = []
    for f in files:
        tmp = pd.read_parquet(f)
        # unify column names if necessary
        if "timestamp_utc" not in tmp.columns and "timestamp" in tmp.columns:
            tmp = tmp.rename(columns={"timestamp": "timestamp_utc"})
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    # parse timestamp
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    # Some APIs use naive timestamps - ensure tz-aware
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    # set index
    df = df.set_index("timestamp_utc").sort_index()
    # keep numeric columns
    numeric_cols = ["pm2_5", "pm10", "o3", "no2", "so2", "co", "temp_c", "humidity", "wind_speed"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    # There may be multiple measurements within same hour; aggregate by mean
    df_hour = df[numeric_cols + ["city"]].groupby([pd.Grouper(freq="1H"), "city"]).mean().reset_index().set_index("timestamp_utc")
    # If multiple cities exist, pick first city per timestamp (for now)
    # If you plan multi-city modeling, change this logic.
    if "city" in df_hour.columns:
        # pivot city -> keep rows where that city is present; for now we'll drop city col and treat all as single-city
        df_hour = df_hour.reset_index().groupby("timestamp_utc").first().set_index("timestamp_utc")
    # Ensure continuous hourly index spanning the min..max
    idx = pd.date_range(start=df_hour.index.min(), end=df_hour.index.max(), freq="1H", tz="UTC")
    df_hour = df_hour.reindex(idx)
    # interpolate numeric columns (time interpolation)
    df_hour[numeric_cols] = df_hour[numeric_cols].interpolate(method="time", limit=6).ffill().bfill()
    return df_hour

def create_features(df):
    df = df.copy()
    # time features (UTC)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month
    # rolling stats for pm2_5
    df["pm2_5_3h_mean"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()
    df["pm2_5_6h_mean"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm2_5_24h_mean"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    # diffs
    df["pm2_5_diff_1h"] = df["pm2_5"].diff(1)
    df["pm2_5_diff_3h"] = df["pm2_5"].diff(3)
    # pct change
    df["pm2_5_pct_1h"] = df["pm2_5"].pct_change(1).fillna(0)
    # lag features
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_3h"] = df["pm2_5"].shift(3)
    # other pollutants rolling means (optional)
    df["pm10_24h_mean"] = df["pm10"].rolling(window=24, min_periods=1).mean()
    # Fill any remaining small NaNs
    df = df.fillna(method="ffill").fillna(method="bfill")
    # create lead targets (future pm2.5)
    for h in [1, 3, 6, 24, 72]:
        df[f"pm2_5_lead_{h}h"] = df["pm2_5"].shift(-h)
    # Drop rows with NaN in any lead target, because those rows do not have full future labels
    lead_cols = [c for c in df.columns if c.startswith("pm2_5_lead_")]
    df = df.dropna(subset=lead_cols)
    return df

def save_features(df):
    out_fname = os.path.join(FEATURE_DIR, f"features_{datetime.now(timezone.utc).date()}.parquet")
    df.to_parquet(out_fname, index=True)
    print("Saved features to", out_fname)
    return out_fname

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-history", type=int, default=48,
                        help="Minimum hours of history required to build features (default 48)")
    args = parser.parse_args(argv)
    df_raw = load_all_raw()
    # Quick sanity check
    if len(df_raw) < args.min_history:
        raise SystemExit(f"Not enough data rows ({len(df_raw)}). Need at least --min-history {args.min_history} hours. Run fetcher with --days larger or use --simulate.")
    features = create_features(df_raw)
    out = save_features(features)
    print("Feature engineering complete. Rows:", len(features))

if __name__ == "__main__":
    main()
