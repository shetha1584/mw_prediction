import requests
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import timedelta
import tempfile
import os

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

# ---------------- CONNECT TO DB ---------------- #
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- GET LATEST PREDICTION DATE ---------------- #
cursor.execute("SELECT MAX(date) FROM prediction;")
latest_pred_date = cursor.fetchone()[0]
if not latest_pred_date:
    raise Exception("No predictions found in the database.")

# 🔹 Use the day BEFORE the latest prediction date
target_date = latest_pred_date - timedelta(days=1)

# ---------------- FETCH PREDICTIONS ---------------- #
cursor.execute("""
    SELECT hour, predicted_kVAh FROM prediction
    WHERE date = %s
    ORDER BY hour
""", (target_date,))
pred_rows = cursor.fetchall()

if not pred_rows:
    raise Exception(f"No prediction data found for {target_date}.")

df_pred = pd.DataFrame(pred_rows, columns=["hour", "predicted_kVAh"])

# ---------------- FETCH ACTUAL DATA FROM API ---------------- #
start_date = f"{target_date}+00:00"
end_date = f"{target_date}+23:59"
url = f"{BASE_URL}?startDateTime={start_date}&endDateTime={end_date}"
response = requests.get(url, headers=HEADERS)
response.raise_for_status()
data = response.json()
actual_data = data.get("consumptionData", {})

if not actual_data:
    raise Exception(f"No actual data returned from API for {target_date}.")

df_actual = pd.DataFrame([
    {"hour": k.split(" ")[1][:5], "actual_kVAh": float(v)}
    for k, v in actual_data.items()
]).sort_values("hour")

# ---------------- MERGE PREDICTED & ACTUAL ---------------- #
df = pd.merge(df_pred, df_actual, on="hour")

# Compute hourly errors
df["abs_error"] = (df["predicted_kVAh"] - df["actual_kVAh"]).abs()
df["percent_error"] = df["abs_error"] / df["actual_kVAh"].replace(0, np.nan) * 100

# Compute accuracy %
MAPE = df["percent_error"].mean()
accuracy_percent = 100 - MAPE

# ---------------- PLOT ---------------- #
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(df["hour"], df["actual_kVAh"], label="Actual", marker="o")
plt.plot(df["hour"], df["predicted_kVAh"], label="Predicted", marker="s")
plt.fill_between(df["hour"], df["actual_kVAh"], df["predicted_kVAh"], color="gray", alpha=0.1)
plt.title(f"Actual vs Predicted Consumption on {target_date}")
plt.xlabel("Hour of Day")
plt.ylabel("Consumption (kVAh)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save figure temporarily
img_path = os.path.join(tempfile.gettempdir(), "actual_vs_pred.png")
plt.savefig(img_path, dpi=300)
plt.close()

# ---------------- BUILD PDF REPORT ---------------- #
pdf_path = f"actual_vs_pred_report_{target_date}.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Graph
story.append(Image(img_path, width=500, height=250))
story.append(Spacer(1, 12))

# Summary
summary = f"""
<b>Consumption Report for {target_date}</b><br/><br/>
<b>Accuracy:</b> {accuracy_percent:.2f}%<br/>
Mean Absolute Error (hourly): {df["abs_error"].mean():.2f} kVAh<br/>
The graph above shows actual vs predicted hourly consumption.
Shaded area represents deviation between prediction and actual values.
"""

story.append(Paragraph(summary, styles["Normal"]))
doc.build(story)

print(f"✅ PDF report saved as: {pdf_path}")

cursor.close()
conn.close()
