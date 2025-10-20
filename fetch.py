import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

# ---------- DATA ---------- #
data_12 = {
  "consumptionData": {
    "2025-10-12 00:00:00": "270.9",
    "2025-10-12 01:00:00": "321.2",
    "2025-10-12 02:00:00": "307.6",
    "2025-10-12 03:00:00": "269.7",
    "2025-10-12 04:00:00": "268.1",
    "2025-10-12 05:00:00": "253.7",
    "2025-10-12 06:00:00": "181.9",
    "2025-10-12 07:00:00": "179.4",
    "2025-10-12 08:00:00": "198.4",
    "2025-10-12 09:00:00": "275.4",
    "2025-10-12 10:00:00": "435.1",
    "2025-10-12 11:00:00": "455.1",
    "2025-10-12 12:00:00": "441.2",
    "2025-10-12 13:00:00": "361.5",
    "2025-10-12 14:00:00": "423.1",
    "2025-10-12 15:00:00": "406.7",
    "2025-10-12 16:00:00": "404.8",
    "2025-10-12 17:00:00": "235.9",
    "2025-10-12 18:00:00": "124.5",
    "2025-10-12 19:00:00": "145.8",
    "2025-10-12 20:00:00": "279.5",
    "2025-10-12 21:00:00": "299.6",
    "2025-10-12 22:00:00": "297.2",
    "2025-10-12 23:00:00": "265.6"
  }
}

data_19 = {
  "consumptionData": {
    "2025-10-19 00:00:00": "290.9",
    "2025-10-19 01:00:00": "297.0",
    "2025-10-19 02:00:00": "240.7",
    "2025-10-19 03:00:00": "173.7",
    "2025-10-19 04:00:00": "116.7",
    "2025-10-19 05:00:00": "34.2",
    "2025-10-19 06:00:00": "20.9",
    "2025-10-19 07:00:00": "26.8",
    "2025-10-19 08:00:00": "26.6",
    "2025-10-19 09:00:00": "25.9",
    "2025-10-19 10:00:00": "57.5",
    "2025-10-19 11:00:00": "58.6",
    "2025-10-19 12:00:00": "56.8",
    "2025-10-19 13:00:00": "38.0",
    "2025-10-19 14:00:00": "66.3",
    "2025-10-19 15:00:00": "44.2",
    "2025-10-19 16:00:00": "19.5",
    "2025-10-19 17:00:00": "0.6",
    "2025-10-19 18:00:00": "0.2",
    "2025-10-19 19:00:00": "0.2",
    "2025-10-19 20:00:00": "0.3",
    "2025-10-19 21:00:00": "2.1",
    "2025-10-19 22:00:00": "10.8",
    "2025-10-19 23:00:00": "0.2"
  }
}

# ---------- PROCESS ---------- #
df_12 = pd.DataFrame({
    "datetime": list(data_12["consumptionData"].keys()),
    "actual": [float(v) for v in data_12["consumptionData"].values()]
})
df_12["datetime"] = pd.to_datetime(df_12["datetime"])
df_12["hour"] = df_12["datetime"].dt.strftime("%H:%M")

df_19 = pd.DataFrame({
    "datetime": list(data_19["consumptionData"].keys()),
    "actual": [float(v) for v in data_19["consumptionData"].values()]
})
df_19["datetime"] = pd.to_datetime(df_19["datetime"])
df_19["hour"] = df_19["datetime"].dt.strftime("%H:%M")

# Compute totals
total_12 = df_12["actual"].sum()
total_19 = df_19["actual"].sum()
percent_diff = ((total_19 - total_12) / total_12) * 100

# ---------- PLOT ---------- #
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(df_12["hour"], df_12["actual"], label="2025-10-12", marker="o")
plt.plot(df_19["hour"], df_19["actual"], label="2025-10-19", marker="s")

plt.fill_between(df_12["hour"], df_12["actual"], df_19["actual"], color="gray", alpha=0.1)
plt.title("Consumption Comparison: 2025-10-12 vs 2025-10-19")
plt.xlabel("Hour of Day")
plt.ylabel("Consumption (kWh)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save image
img_path = os.path.join(tempfile.gettempdir(), "comparison_12_vs_19.png")
plt.savefig(img_path, dpi=300)
plt.close()

# ---------- BUILD PDF ---------- #
pdf_path = "comparison_report_12_vs_19.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Image(img_path, width=500, height=250))
story.append(Spacer(1, 12))

summary = f"""
<b>Comparison Summary: 2025-10-12 vs 2025-10-19</b><br/><br/>
Total Consumption (12 Oct): {total_12:.1f} kWh<br/>
Total Consumption (19 Oct): {total_19:.1f} kWh<br/>
<b>Change:</b> {percent_diff:.1f}% ({'increase' if percent_diff > 0 else 'decrease'})<br/><br/>
19 Oct shows a significant reduction in energy usage across almost all hours, 
especially after 05:00, suggesting minimal or no activity during the evening.<br/>
"""
story.append(Paragraph(summary, styles["Normal"]))
doc.build(story)

print(f"✅ Graph + report saved as: {pdf_path}")
print(f"🔹 2025-10-12 Total: {total_12:.1f} kWh")
print(f"🔹 2025-10-19 Total: {total_19:.1f} kWh")
print(f"📉 Change: {percent_diff:.1f}%")
