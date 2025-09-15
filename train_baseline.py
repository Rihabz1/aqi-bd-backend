import os, joblib
import pandas as pd, numpy as np
import xgboost as xgb

DIVS = ['Dhaka','Chattogram','Sylhet','Khulna','Rajshahi','Barishal','Mymensingh','Rangpur']
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "https://docs.google.com/spreadsheets/d/1aRyCU88momwOk_ONhXXzjbm0-9uoCQrRWVQlQOrTM48/export?format=csv")

def load_and_clean():
    df = pd.read_csv(SHEET_CSV_URL)
    df = df[['Date','City','AQI']].copy()
    # standardize names
    df['City'] = df['City'].replace({'Chittagong':'Chattogram','Chittgong':'Chattogram','Barisal':'Barishal'})
    # keep only 8 divisions
    df = df[df['City'].isin(DIVS)].copy()
    # parse date (your sheet uses MM/DD/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    # numeric AQI; DNA->NaN
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    # per-division interpolation
    df = df.sort_values(['City','Date'])
    df['AQI'] = df.groupby('City', group_keys=False)['AQI'].apply(lambda s: s.interpolate(limit_direction='both'))
    df = df.dropna(subset=['AQI']).reset_index(drop=True)
    return df

def add_features(g, max_lag=14):
    g = g.sort_values('Date').copy()
    for L in range(1, max_lag+1):
        g[f'lag_{L}'] = g['AQI'].shift(L)
    g['roll7']  = g['AQI'].shift(1).rolling(7).mean()
    g['roll14'] = g['AQI'].shift(1).rolling(14).mean()
    g['dow'] = g['Date'].dt.dayofweek
    g['mon'] = g['Date'].dt.month
    g['doy'] = g['Date'].dt.dayofyear
    return g

if __name__ == "__main__":
    df = load_and_clean()
    print(f"Loaded {len(df)} rows from data source")
    parts = [add_features(df[df['City']==d], 14) for d in DIVS]
    feat = pd.concat(parts, ignore_index=True)
    feature_cols = [c for c in feat.columns if c.startswith('lag_')] + ['roll7','roll14','dow','mon','doy']
    feat = feat.dropna(subset=feature_cols+['AQI'])

    os.makedirs("models", exist_ok=True)

    for d in DIVS:
        cdf = feat[feat['City']==d].sort_values('Date')
        if len(cdf) < 120:
            print(f"Skipping {d}, not enough history: {len(cdf)} rows")
            continue
        X, y = cdf[feature_cols], cdf['AQI'].values
        model = xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
            random_state=42
        )
        model.fit(X, y, verbose=False)
        joblib.dump({"model": model, "features": feature_cols}, f"models/{d}.pkl")
        print(f"Saved models/{d}.pkl")

    print("✅ Training done.")
