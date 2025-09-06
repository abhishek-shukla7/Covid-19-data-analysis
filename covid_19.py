# -------------------------------------------------
# Step 1: Import required libraries
# -------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

# -------------------------------------------------
# Step 2: Create a synthetic dataset with regions
# -------------------------------------------------
rng = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
t = np.arange(len(rng))
regions = ["North", "South", "East", "West"]

def gaussian_peak(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

center_2021 = (pd.Timestamp("2021-06-30") - rng[0]).days
np.random.seed(42)

data = []
for r in regions:
    base_amp = np.random.randint(10000, 20000)
    width = np.random.randint(60, 90)
    baseline = np.random.randint(100, 300)
    series = (
        gaussian_peak(t, center_2021, width, base_amp) +
        gaussian_peak(t, center_2021 - 180, width*0.7, base_amp*0.5) +
        gaussian_peak(t, center_2021 + 220, width*0.8, base_amp*0.6) +
        np.random.normal(0, base_amp*0.05, len(t)) +
        baseline
    )
    series = np.maximum(0, series).astype(int)
    for i, d in enumerate(rng):
        data.append([d, r, series[i]])

covid_df = pd.DataFrame(data, columns=["date", "region", "cases"])

# -------------------------------------------------
# Step 3: National summary and peak detection
# -------------------------------------------------
covid_total = covid_df.groupby("date", as_index=False)["cases"].sum()
covid_total["cases_ma7"] = covid_total["cases"].rolling(7, min_periods=1).mean()

peak_row = covid_total.loc[covid_total["cases_ma7"].idxmax()]
print("Peak Date:", peak_row["date"].date())
print("Peak 7-day Avg Cases:", int(peak_row["cases_ma7"]))

# -------------------------------------------------
# Step 4: National Trend Plot
# -------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(covid_total["date"], covid_total["cases"], alpha=0.3, label="Daily Cases")
plt.plot(covid_total["date"], covid_total["cases_ma7"], color="blue", linewidth=2, label="7-day MA")
plt.title("National COVID-19 Daily Cases Trend ", fontsize=14)
plt.xlabel("Date"); plt.ylabel("Cases")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 5: Regional Comparison
# -------------------------------------------------
plt.figure(figsize=(12,6))
for r in regions:
    series = covid_df[covid_df["region"] == r].groupby("date")["cases"].sum().rolling(7, min_periods=1).mean()
    plt.plot(series.index, series.values, label=r)

plt.title("Regional COVID-19 7-day MA Trends", fontsize=14)
plt.xlabel("Date"); plt.ylabel("Cases")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Peak per region
regional_peaks = covid_df.groupby("region")["cases"].max().reset_index()
print("\nRegional Peak Daily Cases:\n", regional_peaks)

# -------------------------------------------------
# Step 6: Interactive Map with Folium
# (Example: plotting regional centers with dummy coords)
# -------------------------------------------------
region_coords = {
    "North": [28.7041, 77.1025],  # Delhi
    "South": [12.9716, 77.5946],  # Bengaluru
    "East": [22.5726, 88.3639],   # Kolkata
    "West": [19.0760, 72.8777]    # Mumbai
}

m = folium.Map(location=[22.0, 80.0], zoom_start=5)

for r in regions:
    peak_val = int(regional_peaks.loc[regional_peaks["region"] == r, "cases"].values[0])
    
    folium.CircleMarker(
        location=region_coords[r],
        radius=10,
        popup=f"{r} Region - Peak Cases: {peak_val}",
        color="red",
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

m.save("covid19_regional_map.html")
print("Interactive map saved as covid19_regional_map.html")
