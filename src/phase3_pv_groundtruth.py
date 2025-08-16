import os
import warnings
import numpy as np
import pandas as pd
import pvlib

warnings.filterwarnings("ignore", category=UserWarning)

WEATHER_PATH = "data/processed/weather_hourly.parquet"
ZONES_PATH   = "data/processed/zones.parquet"
OUT_PATH     = "data/processed/daily_energy.parquet"

SYSTEM_KW_DC = 10.0
SURFACE_TILT = 13.0
SURFACE_AZIM = 180.0
ALBEDO       = 0.20
TZ           = "Asia/Kolkata"

PVWATTS_GAMMA_PDC = -0.004
PVWATTS_INV_EFF   = 0.96

# SAPM cell temperature coefficients (generic open-rack, glass-back)
SAPM_A       = -3.56
SAPM_B       = -0.075
SAPM_DELTA_T = 3.0

def _pvwatts_ac_robust(pdc: pd.Series | np.ndarray, pdc0_w: float, eta_inv_nom: float):
    """
    Robust AC conversion:
      1) try pvlib.inverter.pvwatts (preferred, if present)
      2) fallback: constant-efficiency with clipping to [0, eta*pdc0]
    """
    # Try preferred model from pvlib
    try:
        from pvlib import inverter
        if hasattr(inverter, "pvwatts"):
            return inverter.pvwatts(pdc, pdc0=pdc0_w, eta_inv_nom=eta_inv_nom)
    except Exception:
        pass

    # Fallback: simple constant-efficiency + clipping
    pdc = np.asarray(pdc, dtype=float)
    pac = eta_inv_nom * pdc
    pac = np.clip(pac, 0.0, eta_inv_nom * pdc0_w)
    return pac

def to_ist_index(ts_like) -> tuple[pd.DatetimeIndex, np.ndarray]:
    parsed = pd.to_datetime(ts_like, errors="coerce")
    mask = parsed.notna().to_numpy()
    idx = pd.DatetimeIndex(parsed[mask])
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    else:
        idx = idx.tz_convert(TZ)
    return idx, mask

def pvwatts_daily_kwh(times: pd.DatetimeIndex, ghi_whm2: pd.Series,
                      t2m_c: pd.Series, ws10_ms: pd.Series,
                      lat: float, lon: float) -> pd.Series:
    # Align inputs
    ghi = pd.Series(ghi_whm2.values, index=times).clip(lower=0)
    t2m = pd.Series(t2m_c.values,    index=times)
    ws  = pd.Series(ws10_ms.values,  index=times)

    sol = pvlib.solarposition.get_solarposition(times, lat, lon)
    zen, azm = sol["apparent_zenith"], sol["azimuth"]

    erbs = pvlib.irradiance.erbs(ghi, zen, times)
    dni = erbs["dni"].clip(lower=0)
    dhi = erbs["dhi"].clip(lower=0)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=SURFACE_TILT,
        surface_azimuth=SURFACE_AZIM,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=zen, solar_azimuth=azm,
        albedo=ALBEDO, model="isotropic",
    )
    poa_global = poa["poa_global"].clip(lower=0)

    # SAPM cell temperature with explicit coeffs (fix for earlier error)
    t_cell = pvlib.temperature.sapm_cell(
        poa_global, t2m, ws, a=SAPM_A, b=SAPM_B, deltaT=SAPM_DELTA_T
    )

    pdc0_w = SYSTEM_KW_DC * 1000.0
    pdc = pvlib.pvsystem.pvwatts_dc(
        poa_global, temp_cell=t_cell, pdc0=pdc0_w, gamma_pdc=PVWATTS_GAMMA_PDC
    )

    # Robust AC conversion (works regardless of pvlib version)
    pac = _pvwatts_ac_robust(pdc, pdc0_w=pdc0_w, eta_inv_nom=PVWATTS_INV_EFF)

    pac_w = pd.Series(np.asarray(pac), index=times, name="ac_w")
    daily_kwh = pac_w.resample("D").sum(min_count=20) / 1000.0
    daily_kwh.name = "energy_kwh"
    daily_kwh.index.name = "date"
    return daily_kwh

def main():
    os.makedirs("data/processed", exist_ok=True)
    print("[phase3] loading weather + zones …")

    w = pd.read_parquet(WEATHER_PATH)  # ZoneID, ts, ghi_whm2, t2m_c, ws10_ms
    z = pd.read_parquet(ZONES_PATH)[["ZoneID", "centroid_lat", "centroid_lon"]]

    out = []
    zone_ids = z["ZoneID"].tolist()

    for i, zid in enumerate(zone_ids, 1):
        row = z.loc[z.ZoneID == zid].iloc[0]
        lat = float(row["centroid_lat"])
        lon = float(row["centroid_lon"])

        dfz = w[w.ZoneID == zid][["ts", "ghi_whm2", "t2m_c", "ws10_ms"]].copy()
        if dfz.empty:
            print(f"[phase3][WARN] {zid}: no weather rows, skipping.")
            continue

        times, mask = to_ist_index(dfz["ts"])
        dfz = dfz.loc[mask].copy()
        for col in ["ghi_whm2", "t2m_c", "ws10_ms"]:
            dfz[col] = dfz[col].to_numpy()

        try:
            daily = pvwatts_daily_kwh(times, dfz["ghi_whm2"], dfz["t2m_c"], dfz["ws10_ms"], lat, lon)
            df_daily = daily.reset_index()
            # store naive date (YYYY-MM-DD) for easier parquet joins downstream
            df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.tz_convert(TZ).dt.date
            df_daily.insert(0, "ZoneID", zid)
            out.append(df_daily)
        except Exception as e:
            print(f"[phase3][WARN] {zid}: {e}")

        if i % 25 == 0:
            print(f"[phase3] processed {i}/{len(zone_ids)} zones …")

    if not out:
        raise SystemExit("[phase3] No zones processed; aborting.")

    daily_energy = (
        pd.concat(out, ignore_index=True)
          .dropna(subset=["energy_kwh"])
          .sort_values(["ZoneID", "date"])
          .reset_index(drop=True)
    )

    daily_energy.to_parquet(OUT_PATH)
    print(f"[phase3] wrote {OUT_PATH} with {len(daily_energy):,} rows "
          f"across {daily_energy['ZoneID'].nunique()} zones.")

if __name__ == "__main__":
    main()
