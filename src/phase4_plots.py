# src/phase4_plots.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEAT_PATH = "data/processed/phase4_features.parquet"
CITY_PATH = "data/processed/phase4_city_rollups.parquet"
PLOTS_DIR = "data/plots"

def build_city_rollups_from_features():
    """Create citywide per-day rollups from phase4 features."""
    if not os.path.exists(FEAT_PATH):
        raise FileNotFoundError(f"Missing {FEAT_PATH}. Run phase4_features.py first.")

    fe = pd.read_parquet(FEAT_PATH)
    # Ensure date is datetime (naive is fine for daily aggregations)
    fe["date"] = pd.to_datetime(fe["date"])

    # Basic per-day stats across zones
    g = fe.groupby("date")["energy_kwh"]
    city = pd.DataFrame({
        "n_zones": g.size(),
        "mean_kwh": g.mean(),
        "median_kwh": g.median(),
        "p05_kwh": g.quantile(0.05),
        "p95_kwh": g.quantile(0.95),
        "min_kwh": g.min(),
        "max_kwh": g.max(),
    }).reset_index().sort_values("date")

    # Save for future runs
    os.makedirs(os.path.dirname(CITY_PATH), exist_ok=True)
    city.to_parquet(CITY_PATH, index=False)
    print(f"[plots] built {CITY_PATH} from {FEAT_PATH} ({len(city)} days).")
    return city

def load_city_rollups():
    if os.path.exists(CITY_PATH):
        return pd.read_parquet(CITY_PATH)
    return build_city_rollups_from_features()

def plot_city_timeseries(city: pd.DataFrame):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(12,4))
    plt.plot(city["date"], city["median_kwh"], label="Median")
    # 5–95% band
    plt.fill_between(city["date"], city["p05_kwh"], city["p95_kwh"], alpha=0.2, label="P05–P95")
    plt.title("Citywide PV Daily Energy (10 kW DC) – Median and P05–P95 band")
    plt.ylabel("kWh/day")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "city_daily_band.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[plots] wrote {out}")

def plot_monthly_profile(city: pd.DataFrame):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Month aggregation
    m = (city.assign(month=city["date"].dt.to_period("M").dt.to_timestamp())
              .groupby("month", as_index=False)
              .agg(mean_kwh=("mean_kwh","mean"),
                   median_kwh=("median_kwh","mean"),
                   p05_kwh=("p05_kwh","mean"),
                   p95_kwh=("p95_kwh","mean")))
    plt.figure(figsize=(12,4))
    plt.plot(m["month"], m["median_kwh"], label="Median (avg over months)")
    plt.fill_between(m["month"], m["p05_kwh"], m["p95_kwh"], alpha=0.2, label="P05–P95 (avg)")
    plt.title("Citywide Monthly Profile – Avg of Daily Stats by Month")
    plt.ylabel("kWh/day")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, "city_monthly_profile.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[plots] wrote {out}")

def main():
    city = load_city_rollups()
    # Basic sanity print
    print(f"[plots] days in city rollups: {city['date'].nunique()} | "
          f"range: {city['date'].min().date()} → {city['date'].max().date()}")

    plot_city_timeseries(city)
    plot_monthly_profile(city)

if __name__ == "__main__":
    main()
