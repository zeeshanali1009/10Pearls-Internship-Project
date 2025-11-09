# train_model.py
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
TRAINING_TABLE = os.path.join(ROOT, 'training_data.parquet')

FEATURE_COLS = [
    'pm2_5','pm10','so2','no2','o3','co',
    'hour','day','weekday','month',
    'pm25_lag1','pm25_rolling3','pm25_change_rate'
]

if __name__ == '__main__':
    df = pd.read_parquet(TRAINING_TABLE)
    df = df.dropna(subset=FEATURE_COLS + ['target_aqi'])

    X = df[FEATURE_COLS].fillna(0)
    y = df['target_aqi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f'RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}')

    v = 1
    model_path = os.path.join(MODEL_DIR, f'rf_aqi_v{v}.joblib')
    joblib.dump({'model': model, 'features': FEATURE_COLS}, model_path)
    print('Saved model to', model_path)

    fi = pd.DataFrame({ 'feature': FEATURE_COLS, 'importance': model.feature_importances_ })
    fi.sort_values('importance', ascending=False).to_csv(os.path.join(MODEL_DIR,'feature_importances.csv'), index=False)
