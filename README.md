# Pearls AQI Predictor

Minimal end-to-end AQI prediction system (fetch -> features -> train -> serve).

## Quick start
1. Create a Python 3.9+ virtualenv
2. `pip install -r requirements.txt`
3. Put your OpenWeather API key in `config.json` or as environment variable `OPENWEATHER_API_KEY`.
4. Run `python scripts/fetch_features.py --start 2025-01-01 --end 2025-11-01 --city "Lahore,PK"` to backfill historical hourly data.
5. Run `python scripts/feature_engineering.py` to build the training table.
6. Run `python scripts/train_model.py` to train and save the model.
7. Run the Streamlit app:
   ```
   streamlit run app_streamlit.py
   ```

Files:
- scripts/fetch_features.py: fetch historical and forecast air-pollution + weather
- scripts/feature_engineering.py: compute time-based & derived features
- scripts/train_model.py: trains a RandomForest regressor and saves model
- app_streamlit.py: small dashboard to show predictions for next 72 hours
- requirements.txt: Python deps

Notes:
- This repo uses local parquet files as a feature store (folder `feature_store/`).
- The code computes target AQI from PM2.5 using US EPA breakpoints. If you have official AQI values in dataset, you may use those instead.
