import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from xgboost import XGBRegressor
import psycopg2
from psycopg2.extras import execute_values
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- CONFIG ---------------- #
BASE_URL = "https://mw.elementsenergies.com/api/hkVAhconsumptiontest"
HEADERS = {
    "Authorization": "Bearer 9cF7Gk2MZpQ8XvT5LbR3NdYqWjK6HsA4"
}

DB_CONFIG = {
    "host": "localhost",
    "dbname": "test_elements;",
    "user": "postgres",
    "password": "ABcd1234!@",
    "port": 5432
}

IDLE_LOAD = 0
PRED_FLOOR = 0.0
PRED_CAP = 650.0
SHUTDOWN_CAP = 15.0   # holiday + shutdown cap

# ---------------- CONNECT ---------------- #
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- STEP 1: LOAD DATA ---------------- #
cursor.execute("SELECT date, hour, kvah FROM consumption_data ORDER BY date, hour;")
df = pd.DataFrame(cursor.fetchall(), columns=["date", "hour", "y"])

df["hour"] = df["hour"].str[:5]
df["ds"] = pd.to_datetime(df["date"] + " " + df["hour"])
df = df[["ds", "y"]].drop_duplicates("ds").sort_values("ds")

# ---------------- STEP 2: HOLIDAYS ---------------- #
cursor.execute("SELECT date FROM holidays;")
holiday_dates = pd.to_datetime([r[0] for r in cursor.fetchall()]).date

# ---------------- STEP 3: FEATURES ---------------- #
df["hour"] = df["ds"].dt.hour
df["day_of_week"] = df["ds"].dt.weekday

# Shutdown window ONLY
def is_shutdown(row):
    if row["day_of_week"] == 6 and row["hour"] >= 20:
        return 1
    if row["day_of_week"] == 0 and row["hour"] < 5:
        return 1
    return 0

df["is_shutdown"] = df.apply(is_shutdown, axis=1)

# Holidays
df["is_holiday"] = df["ds"].dt.date.isin(holiday_dates).astype(int)
df["is_day_after_holiday"] = df["ds"].dt.date.shift(1).isin(holiday_dates).astype(int)

# Shift (descriptive only)
df["shift_flag"] = np.where((df["hour"] >= 8) & (df["hour"] < 20), 1, 2)

# Time features
df["is_lunch_hour"] = (df["hour"] == 13).astype(int)
df["is_morning_dip"] = df["hour"].isin([5, 6, 7]).astype(int)
df["is_shift_change"] = df["hour"].isin([8, 20]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["cap"] = PRED_CAP
df["floor"] = PRED_FLOOR
df["hour_of_day"] = df["hour"]

# ---------------- STEP 4: PROPHET ---------------- #
m = Prophet(
    growth="logistic",
    weekly_seasonality=True,
    daily_seasonality=False,
    yearly_seasonality=False,
    seasonality_mode="additive"
)

m.add_seasonality("daily_24h", 24, 8)

for reg in [
    "hour_of_day",
    "is_lunch_hour",
    "is_morning_dip",
    "is_shift_change"
]:
    m.add_regressor(reg)

train_cols = [
    "ds", "y", "cap", "floor",
    "hour_of_day",
    "is_lunch_hour",
    "is_morning_dip",
    "is_shift_change"
]

m.fit(df[train_cols])

# ---------------- STEP 5: XGBOOST RESIDUALS ---------------- #
pred_hist = m.predict(df[train_cols])["yhat"]
df["residual"] = df["y"] - pred_hist
df["lag_1h"] = df["y"].shift(1)
df["lag_24h"] = df["y"].shift(24)

df_ml = df.dropna()

X = df_ml[
    ["hour", "day_of_week", "shift_flag",
     "lag_1h", "lag_24h",
     "hour_sin", "hour_cos",
     "is_lunch_hour", "is_morning_dip", "is_shift_change"]
]

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X, df_ml["residual"])

# ---------------- STEP 6: PREDICT NEXT DAY ---------------- #
next_day = df["ds"].max().date() + timedelta(days=1)
future_df = pd.DataFrame({"ds": pd.date_range(next_day, periods=24, freq="H")})

future_df["hour"] = future_df["ds"].dt.hour
future_df["day_of_week"] = future_df["ds"].dt.weekday
future_df["shift_flag"] = np.where((future_df["hour"] >= 8) & (future_df["hour"] < 20), 1, 2)
future_df["is_lunch_hour"] = (future_df["hour"] == 13).astype(int)
future_df["is_morning_dip"] = future_df["hour"].isin([5, 6, 7]).astype(int)
future_df["is_shift_change"] = future_df["hour"].isin([8, 20]).astype(int)
future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
future_df["hour_of_day"] = future_df["hour"]
future_df["cap"] = PRED_CAP
future_df["floor"] = PRED_FLOOR

future_df["is_shutdown"] = future_df.apply(is_shutdown, axis=1)
future_df["is_holiday"] = future_df["ds"].dt.date.isin(holiday_dates).astype(int)

future_df["prophet_pred"] = m.predict(future_df[train_cols])["yhat"]

# XGB correction
hist = df.set_index("ds")
final_preds = []

for i, row in future_df.iterrows():
    lag1 = final_preds[-1] if i > 0 else hist["y"].get(row["ds"] - timedelta(hours=1))
    lag24 = hist["y"].get(row["ds"] - timedelta(days=1))

    if pd.isna(lag1) or pd.isna(lag24):
        final_preds.append(row["prophet_pred"])
        continue

    feats = row[X.columns].copy()
    feats["lag_1h"] = lag1
    feats["lag_24h"] = lag24

    resid = xgb.predict(pd.DataFrame([feats]))[0]
    final_preds.append(row["prophet_pred"] + resid)

future_df["predicted_kVAh"] = np.clip(final_preds, PRED_FLOOR, PRED_CAP)

# ---------------- STEP 7: HARD FORCING ---------------- #
mask = (future_df["is_shutdown"] == 1) | (future_df["is_holiday"] == 1)
future_df.loc[mask, "predicted_kVAh"] = np.minimum(
    future_df.loc[mask, "predicted_kVAh"],
    SHUTDOWN_CAP
)

future_df["predicted_kVAh"] = future_df["predicted_kVAh"].round(2)

# ---------------- STEP 8: SAVE ---------------- #
future_df["date"] = future_df["ds"].dt.date
future_df["hour"] = future_df["ds"].dt.strftime("%H:%M")

execute_values(
    cursor,
    "INSERT INTO prediction (date, hour, predicted_kVAh) VALUES %s",
    list(future_df[["date", "hour", "predicted_kVAh"]].itertuples(index=False, name=None))
)

conn.commit()
cursor.close()
conn.close()
