import os, time, joblib
import pandas as pd, numpy as np
from datetime import timedelta
from fastapi import FastAPI, HTTPException

DIVS = ['Dhaka','Chattogram','Sylhet','Khulna','Rajshahi','Barishal','Mymensingh','Rangpur']
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "https://docs.google.com/spreadsheets/d/1aRyCU88momwOk_ONhXXzjbm0-9uoCQrRWVQlQOrTM48/export?format=csv")
CACHE_SECS = int(os.getenv("CACHE_SECS", "1800"))  # 30m default

# load all per-division models
MODELS, FEATURES = {}, None
for d in DIVS:
    p = f"models/{d}.pkl"
    if os.path.exists(p):
        obj = joblib.load(p)
        MODELS[d] = obj['model']
        FEATURES = obj['features']

if not MODELS:
    raise RuntimeError("No models found in models/")

# cache for sheet data
_DATA = {"df": None, "ts": 0}

def load_sheet_clean():
    if not SHEET_CSV_URL:
        raise RuntimeError("SHEET_CSV_URL not set")
    df = pd.read_csv(SHEET_CSV_URL)
    df = df[['Date','City','AQI']].copy()
    df['City'] = df['City'].replace({'Chittagong':'Chattogram','Chittgong':'Chattogram','Barisal':'Barishal'})
    df = df[df['City'].isin(DIVS)].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df = df.sort_values(['City','Date'])
    df['AQI'] = df.groupby('City', group_keys=False)['AQI'].apply(lambda s: s.interpolate(limit_direction='both'))
    return df.dropna(subset=['AQI']).reset_index(drop=True)

def get_data_cached():
    now = time.time()
    if _DATA["df"] is None or (now - _DATA["ts"] > CACHE_SECS):
        _DATA["df"] = load_sheet_clean()
        _DATA["ts"] = now
    return _DATA["df"]

def build_feats(work):
    work = work.copy().sort_values('Date')
    for L in range(1, 15):
        work[f'lag_{L}'] = work['AQI'].shift(L)
    work['roll7']  = work['AQI'].shift(1).rolling(7).mean()
    work['roll14'] = work['AQI'].shift(1).rolling(14).mean()
    work['dow'] = work['Date'].dt.dayofweek
    work['mon'] = work['Date'].dt.month
    work['doy'] = work['Date'].dt.dayofyear
    return work

app = FastAPI(title="BD AQI Forecast API", version="1.0")

@app.get("/health")
def health():
    return {"status":"ok","divisions":list(MODELS.keys())}

@app.get("/predict")
def predict(division: str):
    division = division.strip()
    if division not in DIVS:
        raise HTTPException(400, detail=f"division must be one of {DIVS}")
    if division not in MODELS:
        raise HTTPException(500, detail=f"No model for {division}")

    model = MODELS[division]
    df_all = get_data_cached()
    hist = df_all[df_all["City"]==division].sort_values("Date")[["Date","AQI"]].copy()
    if hist.empty:
        raise HTTPException(404, detail=f"No data for {division}")

    last_date = hist["Date"].max()
    work = hist.copy()
    out = []

    for step in range(1, 8):
        target_date = last_date + timedelta(days=step)
        tmp = build_feats(work)
        x = tmp.iloc[-1:][FEATURES]
        if x.isna().any(axis=None):
            pred = float(work["AQI"].tail(7).mean())
        else:
            pred = float(model.predict(x)[0])
        out.append({"date": target_date.date().isoformat(), "predicted_aqi": pred})
        work = pd.concat([work, pd.DataFrame([{"Date": target_date, "AQI": pred}])], ignore_index=True)

    return {"division": division, "forecast_days": 7, "predictions": out}
