import os, time, requests, pytz, numpy as np, pandas as pd
from datetime import datetime, timedelta
from typing import Dict

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
TZ = "Asia/Kolkata"
START_DATE = "2023-01-01"          # extend later if you want
RAW_DIR = "data/raw/weather"       # per-zone cache for resume/debug
OUT_PATH = "data/processed/weather_hourly.parquet"
PARAMS = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M"]  # GHI, temp, wind@10m

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def _end_date_utc_minus1() -> str:
    return (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

def fetch_power_hourly(lat: float, lon: float, start: str, end: str, max_retries=3, sleep=0.6) -> pd.DataFrame:
    params = {
        "parameters": ",".join(PARAMS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(POWER_URL, params=params, timeout=60)
            r.raise_for_status()
            j = r.json()
            series = j.get("properties", {}).get("parameter", {})
            if not series:
                raise RuntimeError("No POWER data returned")
            def expand(var):
                s = pd.Series(series[var], name=var)
                idx = pd.to_datetime(s.index, format="%Y%m%d%H", utc=True)
                idx = idx.tz_convert(TZ).tz_localize(None)  # IST, naive
                s.index = idx
                return s
            df = pd.concat([expand(v) for v in PARAMS], axis=1)
            df.columns = ["ghi_wm2", "t2m_c", "ws10_ms"]
            return df
        except Exception as e:
            if attempt == max_retries: raise
            time.sleep(sleep * attempt)

def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().asfreq("H")
    # small gaps (<=3h) ffill
    small_gap = df.isna().rolling(3).sum().fillna(0) <= 3
    df = df.ffill().where(small_gap)
    # night/negatives → 0
    df["ghi_wm2"] = df["ghi_wm2"].clip(lower=0)
    # hourly Wh/m² (numerically same at 1h step)
    df["ghi_whm2"] = df["ghi_wm2"] * 1.0
    df = df.drop(columns=["ghi_wm2"])
    # robust clip
    for col in ["ghi_whm2", "t2m_c", "ws10_ms"]:
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q99)
    return df

def process_zone(zrow: Dict, start: str, end: str) -> pd.DataFrame:
    zid = zrow["ZoneID"]; lat = float(zrow["centroid_lat"]); lon = float(zrow["centroid_lon"])
    raw_fp = os.path.join(RAW_DIR, f"{zid}.parquet")

    if os.path.exists(raw_fp):
        raw = pd.read_parquet(raw_fp).set_index("ts")
        raw.index = pd.to_datetime(raw.index)
    else:
        raw = fetch_power_hourly(lat, lon, start, end)
        tmp = raw.copy(); tmp["ts"] = tmp.index
        tmp.to_parquet(raw_fp)
        time.sleep(0.2)  # be polite

    clean = clean_hourly(raw).reset_index().rename(columns={"index": "ts"})
    clean.insert(0, "ZoneID", zid)
    return clean[["ZoneID", "ts", "ghi_whm2", "t2m_c", "ws10_ms"]]

def main():
    zones = pd.read_parquet("data/processed/zones.parquet")
    start, end = START_DATE, _end_date_utc_minus1()

    frames = []
    for i, row in zones.iterrows():
        try:
            dfz = process_zone(row, start, end)
            frames.append(dfz)
            if (i + 1) % 25 == 0:
                print(f"[phase2] processed {i+1}/{len(zones)} zones …")
        except Exception as e:
            print(f"[phase2][WARN] {row['ZoneID']}: {e}")

    if not frames:
        raise SystemExit("No zones processed; abort.")
    weather = pd.concat(frames, ignore_index=True)
    weather["ts"] = pd.to_datetime(weather["ts"])
    weather = weather.sort_values(["ZoneID","ts"]).reset_index(drop=True)
    weather.to_parquet(OUT_PATH)
    print(f"[phase2] wrote {OUT_PATH} with {len(weather):,} rows across {weather['ZoneID'].nunique()} zones. "
          f"Range: {weather['ts'].min()} → {weather['ts'].max()}")

if __name__ == "__main__":
    main()
