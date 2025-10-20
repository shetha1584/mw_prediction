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

# ---------------- CONNECT TO DB ---------------- #
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- STEP 1: Fetch latest consumption data ---------------- #
cursor.execute("SELECT MAX(date + hour::interval) FROM consumption_data;")
latest_ts = cursor.fetchone()[0]

yesterday = datetime.now() - timedelta(days=1)
end_date_obj = yesterday.replace(hour=23, minute=59, second=0, microsecond=0)

if latest_ts and latest_ts >= end_date_obj:
    print("Data is already up-to-date ✅")
else:
    start_date_obj = latest_ts + timedelta(hours=1) if latest_ts else datetime.strptime("2025-04-01 00:00", "%Y-%m-%d %H:%M")
    start_date = start_date_obj.strftime("%Y-%m-%d+%H:%M")
    end_date = end_date_obj.strftime("%Y-%m-%d+%H:%M")
    print(f"Fetching data from {start_date} to {end_date}")

    url = f"{BASE_URL}?startDateTime={start_date}&endDateTime={end_date}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    consumption_data = data.get("consumptionData", {})

    if consumption_data:
        records = []
        for timestamp, value in consumption_data.items():
            date_str, time_str = timestamp.split(" ")
            records.append({
                "datetime": pd.to_datetime(timestamp),
                "date": date_str,
                "hour": time_str,
                "kvah": float(value)
            })
        df_new = pd.DataFrame(records).set_index("datetime").sort_index()

        # ---- CLEAN DATA ---- #
        def clean_factory_data(df, freq="H", idle_load=IDLE_LOAD):
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            df = df.reindex(full_range)

            mask_off = (
                ((df.index.weekday == 6) & (df.index.hour >= 8)) |
                ((df.index.weekday == 0) & (df.index.hour < 8))
            )
            df.loc[mask_off, "kvah"] = df.loc[mask_off, "kvah"].fillna(idle_load)
            df["kvah"] = df["kvah"].interpolate(method="time")
            df["hour"] = df.index.hour
            df["shift"] = np.where((df["hour"] >= 8) & (df["hour"] < 20), "day", "night")
            df["kvah"] = df.groupby("shift")["kvah"].transform(lambda x: x.fillna(x.mean()))
            return df.drop(columns=["hour", "shift"])

        df_new = clean_factory_data(df_new)

        df_new["date"] = df_new.index.strftime("%Y-%m-%d")
        df_new["hour"] = df_new.index.strftime("%H:%M:%S")
        values = [(row["date"], row["hour"], row["kvah"]) for _, row in df_new.iterrows()]

        insert_query = """
            INSERT INTO consumption_data (date, hour, kvah)
            VALUES %s
            ON CONFLICT (date, hour) DO UPDATE SET kvah = EXCLUDED.kvah;
        """
        execute_values(cursor, insert_query, values)
        conn.commit()
        print(f"Inserted/Updated {len(values)} rows into consumption_data ✅")

# ---------------- STEP 2: Load full historical data ---------------- #
cursor.execute("SELECT date, hour, kvah FROM consumption_data ORDER BY date, hour;")
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=["date", "hour", "y"])

df["hour"] = df["hour"].astype(str).str.slice(0, 5)
df["ds"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"])
df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
df = df.drop_duplicates(subset="ds", keep="first")

# ---------------- STEP 3: Calendar & Sunday off features ---------------- #
cursor.execute("SELECT date, description FROM holidays;")
holiday_rows = cursor.fetchall()
holidays_df = pd.DataFrame(holiday_rows, columns=["ds", "holiday"])
holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

sundays_off = []
if len(df) > 0:
    start_hist = df["ds"].min().floor("D")
    end_hist = (df["ds"].max() + pd.Timedelta(days=7)).ceil("D")
    for d in pd.date_range(start_hist, end_hist, freq="W-SUN"):
        start = d + pd.Timedelta(hours=8)
        end = d + pd.Timedelta(days=1, hours=8)
        hours_range = pd.date_range(start, end, freq="H", inclusive="left")
        for hr in hours_range:
            sundays_off.append({"ds": hr, "holiday": "sunday_off"})

sunday_df = pd.DataFrame(sundays_off)
holidays_all = pd.concat([holidays_df, sunday_df], ignore_index=True).drop_duplicates(subset=["ds", "holiday"])

# ---------------- STEP 4: Operational features ---------------- #
df["hour"] = df["ds"].dt.hour
df["day_of_week"] = df["ds"].dt.weekday
df["is_sunday"] = (df["day_of_week"] == 6).astype(int)

def get_shift(row):
    if row["is_sunday"] == 1 and row["hour"] >= 8:
        return 0
    if row["is_sunday"] == 1 and row["hour"] < 8:
        return 2
    if 8 <= row["hour"] < 20:
        return 1
    return 2

df["shift_flag"] = df.apply(get_shift, axis=1)
df["is_lunch_hour"] = (df["hour"] == 13).astype(int)
df["is_morning_dip"] = df["hour"].isin([5, 6, 7]).astype(int)
df["is_shift_change"] = df["hour"].isin([8, 20]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["hour_of_day"] = df["hour"]
df["cap"] = PRED_CAP
df["floor"] = PRED_FLOOR

# ---------------- STEP 4b: Special day effects ---------------- #
df["is_holiday"] = df["ds"].isin(holidays_df["ds"]).astype(int)
holiday_dates = holidays_df["ds"].dt.date.tolist()
df["day_after_holiday"] = df["ds"].dt.date.shift(-1).isin(holiday_dates).astype(int)
df["is_sunday_special"] = ((df["day_of_week"] == 6) & (df["y"] >= 50) & (df["hour"] >= 7)).astype(int)
df["is_monday_special"] = (df["day_of_week"] == 0) | df["day_after_holiday"]
df["sunday_effect"] = df["is_sunday_special"]
df["monday_effect"] = df["is_monday_special"]
df["holiday_effect"] = df["is_holiday"]

# ---------------- STEP 5: Train Prophet ---------------- #
m = Prophet(
    growth="logistic",
    weekly_seasonality=True,
    daily_seasonality=False,
    yearly_seasonality=False,
    holidays=holidays_all,
    seasonality_mode="additive"
)

m.add_seasonality(name="daily_24h", period=24, fourier_order=8)
m.add_seasonality(name="shift_12h", period=12, fourier_order=5)
m.add_seasonality(name="shift_cycle", period=12, fourier_order=6)
m.add_regressor("hour_of_day")
m.add_regressor("is_lunch_hour")
m.add_regressor("is_morning_dip")
m.add_regressor("is_shift_change")
m.add_regressor("sunday_effect")
m.add_regressor("monday_effect")
m.add_regressor("holiday_effect")

prophet_train_cols = ["ds", "y", "cap", "floor",
                      "hour_of_day", "is_lunch_hour", "is_morning_dip", "is_shift_change",
                      "sunday_effect", "monday_effect", "holiday_effect"]
m.fit(df[prophet_train_cols])

# ---------------- STEP 6: Residuals & XGBoost ---------------- #
forecast_history = m.predict(df[prophet_train_cols])
df["prophet_pred"] = forecast_history["yhat"].values
df["residual"] = df["y"] - df["prophet_pred"]
df["lag_1h"] = df["y"].shift(1)
df["lag_24h"] = df["y"].shift(24)
df_ml = df.dropna().copy()

xgb_features = [
    "hour", "day_of_week", "shift_flag", "is_sunday",
    "lag_1h", "lag_24h",
    "hour_sin", "hour_cos",
    "is_lunch_hour", "is_morning_dip", "is_shift_change"
]

X = df_ml[xgb_features]
y_resid = df_ml["residual"]

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)
xgb_model.fit(X, y_resid)

# ---------------- STEP 7: Predict Next Day ---------------- #
last_date = df["ds"].max().date()
next_date = last_date + timedelta(days=1)
future_dates = pd.date_range(start=datetime.combine(next_date, datetime.min.time()), periods=24, freq="H")
future_df = pd.DataFrame({"ds": future_dates})
future_df["hour"] = future_df["ds"].dt.hour
future_df["day_of_week"] = future_df["ds"].dt.weekday
future_df["is_sunday"] = (future_df["day_of_week"] == 6).astype(int)
future_df["shift_flag"] = future_df.apply(get_shift, axis=1)
future_df["is_lunch_hour"] = (future_df["hour"] == 13).astype(int)
future_df["is_morning_dip"] = future_df["hour"].isin([5, 6, 7]).astype(int)
future_df["is_shift_change"] = future_df["hour"].isin([8, 20]).astype(int)
future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24.0)
future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24.0)
future_df["hour_of_day"] = future_df["hour"]
future_df["cap"] = PRED_CAP
future_df["floor"] = PRED_FLOOR

# Add special day effects for future
future_df["is_holiday"] = future_df["ds"].isin(holidays_df["ds"]).astype(int)
future_df["day_after_holiday"] = future_df["ds"].dt.date.shift(-1).isin(holiday_dates).astype(int)
future_df["is_sunday_special"] = ((future_df["day_of_week"] == 6) & (future_df["hour"] >= 7)).astype(int)
future_df["is_monday_special"] = (future_df["day_of_week"] == 0) | future_df["day_after_holiday"]
future_df["sunday_effect"] = future_df["is_sunday_special"]
future_df["monday_effect"] = future_df["is_monday_special"]
future_df["holiday_effect"] = future_df["is_holiday"]

prophet_future_cols = ["ds", "cap", "floor",
                       "hour_of_day", "is_lunch_hour", "is_morning_dip", "is_shift_change",
                       "sunday_effect", "monday_effect", "holiday_effect"]
prophet_forecast = m.predict(future_df[prophet_future_cols])
future_df["prophet_pred"] = prophet_forecast["yhat"].values

# Gradual Sunday rise/drop logic
for i, row in future_df.iterrows():
    if row["sunday_effect"]:
        if 7 <= row["hour"] < 15:
            future_df.at[i, "prophet_pred"] = max(row["prophet_pred"], 250)
        elif 15 <= row["hour"] <= 17:
            drop_factor = (17 - row["hour"]) / 2
            future_df.at[i, "prophet_pred"] = row["prophet_pred"] * drop_factor

# Residual XGBoost correction
hist = df.set_index("ds")
def safe_lookup(dt, hours=0, days=0, default=np.nan):
    key = dt - pd.Timedelta(hours=hours) - pd.Timedelta(days=days)
    try:
        return float(hist.loc[key, "y"])
    except Exception:
        return default

final_preds = []
lag1_list, lag24_list = [], []

for i, dt in enumerate(future_df["ds"]):
    lag24 = safe_lookup(dt, days=1)
    lag1 = final_preds[-1] if i > 0 else safe_lookup(dt, hours=1)
    row = future_df.iloc[i]
    feats = {f: row[f] for f in ["hour", "day_of_week", "shift_flag", "is_sunday",
                                 "hour_sin", "hour_cos", "is_lunch_hour", "is_morning_dip", "is_shift_change"]}
    feats["lag_1h"], feats["lag_24h"] = lag1, lag24
    feats_df = pd.DataFrame([feats], columns=xgb_features)

    resid_correction = 0.0 if np.isnan(lag1) or np.isnan(lag24) else float(xgb_model.predict(feats_df)[0])
    pred = np.clip(row["prophet_pred"] + resid_correction, PRED_FLOOR, PRED_CAP)
    final_preds.append(pred)
    lag1_list.append(lag1)
    lag24_list.append(lag24)

future_df["lag_1h"] = lag1_list
future_df["lag_24h"] = lag24_list
future_df["predicted_kVAh"] = np.round(final_preds, 2)

# ---------------- STEP 8: Save Predictions ---------------- #
future_df["date"] = future_df["ds"].dt.date
future_df["hour_str"] = future_df["ds"].dt.strftime("%H:%M")
pred_result = future_df[["date", "hour_str", "predicted_kVAh"]].rename(columns={"hour_str": "hour"})

cursor.execute("DROP TABLE IF EXISTS prediction;")
cursor.execute("""
CREATE TABLE prediction (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hour VARCHAR(5) NOT NULL,
    predicted_kVAh DOUBLE PRECISION NOT NULL
);
""")
conn.commit()

values = list(pred_result.itertuples(index=False, name=None))
insert_query = "INSERT INTO prediction (date, hour, predicted_kVAh) VALUES %s;"
execute_values(cursor, insert_query, values)
conn.commit()

# Append to prediction history
overwrite_history = False  # Set True if you want to overwrite history
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hour VARCHAR(5) NOT NULL,
    predicted_kVAh DOUBLE PRECISION NOT NULL,
    UNIQUE(date, hour)
);
""")
conn.commit()

if overwrite_history:
    insert_hist_query = """
    INSERT INTO prediction_history (date, hour, predicted_kVAh)
    VALUES %s
    ON CONFLICT (date, hour) DO UPDATE SET predicted_kVAh = EXCLUDED.predicted_kVAh;
    """
else:
    insert_hist_query = """
    INSERT INTO prediction_history (date, hour, predicted_kVAh)
    VALUES %s
    ON CONFLICT (date, hour) DO NOTHING;
    """
execute_values(cursor, insert_hist_query, values)
conn.commit()

print(f"Inserted {len(values)} rows into 'prediction' table (latest forecast)")
print(f"Updated {len(values)} rows into 'prediction_history' table (history)")

cursor.close()
conn.close()