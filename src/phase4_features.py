#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

DAILY_PATH          = "data/processed/daily_energy.parquet"       # from Phase 3
QA_SUMMARY_CSV      = "data/processed/phase4_qaqc_summary.csv"    # zone-level summary (preferred)
QA_PERYEAR_PARQUET  = "data/processed/phase4_qaqc.parquet"        # per zone-year fallback
FEATURES_PATH       = "data/processed/phase4_features.parquet"

def get_good_zones() -> set[str]:
    """
    Determine the set of ZoneID that pass QA.
    1) Prefer zone-level summary CSV with a column `qa_pass_zone` (bool/1/0/yes/no/True/False)
    2) Fallback to per-year parquet; consider a zone passing if *any* full year passes.
    """
    # Preferred: zone-level summary CSV
    if os.path.exists(QA_SUMMARY_CSV):
        qa = pd.read_csv(QA_SUMMARY_CSV)
        if "qa_pass_zone" not in qa.columns:
            # Backward compatibility: look for older column names
            for alt in ("zone_pass", "qa_pass", "qa_pass_diag"):
                if alt in qa.columns:
                    qa = qa.rename(columns={alt: "qa_pass_zone"})
                    break
        if "qa_pass_zone" in qa.columns:
            flag = qa["qa_pass_zone"]
            if flag.dtype != bool:
                flag = flag.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
            return set(qa.loc[flag, "ZoneID"])

    # Fallback: derive from per-year parquet
    if os.path.exists(QA_PERYEAR_PARQUET):
        qy = pd.read_parquet(QA_PERYEAR_PARQUET)
        # prefer a boolean pass flag if present
        pass_col = None
        for c in qy.columns:
            cl = c.lower()
            if cl in {"qa_pass", "qa_pass_diag", "qa"}:
                pass_col = c
                break

        if pass_col is not None:
            # consider a "full" year as days >= 360 if present
            days_col = next((c for c in qy.columns if c.lower() == "days"), None)
            qy_full = qy if days_col is None else qy[qy[days_col] >= 360]
            # zone passes if ANY full-year pass is True
            zpass = qy_full.groupby("ZoneID")[pass_col].any().reset_index(name="zone_pass")
            return set(zpass.loc[zpass["zone_pass"], "ZoneID"])

        # If no pass flag available, allow all zones present
        return set(qy["ZoneID"].unique())

    # Nothing found -> empty set
    return set()

def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily with columns [ZoneID, date, energy_kwh]
    Output: features DataFrame at daily grain with:
      - calendar: year, month, doy, dow, is_weekend
      - rolling stats: roll7_mean/std/z; roll30_mean
      - monthly climatology (per zone-month across years) and anomaly
    """
    df = daily.copy()

    # Basic hygiene
    if "date" not in df.columns or "ZoneID" not in df.columns or "energy_kwh" not in df.columns:
        raise ValueError("daily_energy must have columns: ZoneID, date, energy_kwh")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "energy_kwh"]).sort_values(["ZoneID", "date"]).reset_index(drop=True)

    # Calendar fields
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["doy"]        = df["date"].dt.dayofyear
    df["dow"]        = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6])

    # Rolling stats per zone (requires a time-sorted index)
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        s = g["energy_kwh"]
        g["roll7_mean"]  = s.rolling(7, min_periods=3).mean()
        g["roll7_std"]   = s.rolling(7, min_periods=3).std(ddof=0)
        g["roll7_z"]     = (s - g["roll7_mean"]) / g["roll7_std"].replace(0, np.nan)
        g["roll30_mean"] = s.rolling(30, min_periods=10).mean()
        return g

    df = df.groupby("ZoneID", group_keys=False, sort=False).apply(_roll)

    # Monthly climatology per zone
    clim = (df.groupby(["ZoneID", "month"], as_index=False)["energy_kwh"]
              .mean()
              .rename(columns={"energy_kwh": "clim_month_kwh"}))

    df = df.merge(clim, on=["ZoneID", "month"], how="left")
    df["anom_month_kwh"] = df["energy_kwh"] - df["clim_month_kwh"]

    # Tidy column order
    cols = [
        "ZoneID", "date", "energy_kwh",
        "year", "month", "doy", "dow", "is_weekend",
        "roll7_mean", "roll7_std", "roll7_z", "roll30_mean",
        "clim_month_kwh", "anom_month_kwh",
    ]
    return df[cols]

def main():
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    if not os.path.exists(DAILY_PATH):
        print(f"[features][ERR] Missing {DAILY_PATH}. Run Phase 3 first.", file=sys.stderr)
        sys.exit(2)

    # Load daily energy (Phase 3)
    de = pd.read_parquet(DAILY_PATH)
    # Ensure expected schema
    missing = {"ZoneID", "date", "energy_kwh"} - set(de.columns)
    if missing:
        print(f"[features][ERR] {DAILY_PATH} missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(2)

    # Determine QA-passing zones
    good = get_good_zones()
    if len(good) == 0:
        print("[features][WARN] No QA list found or no zones passed; proceeding with ALL zones.")
        good = set(de["ZoneID"].unique())

    de = de[de["ZoneID"].isin(good)].copy()
    print(f"[features] zones passing QA: {de['ZoneID'].nunique()}")

    # Build features & save
    feats = build_features(de)
    feats.to_parquet(FEATURES_PATH, index=False)
    print(f"[features] wrote {FEATURES_PATH} with {len(feats):,} rows")

if __name__ == "__main__":
    main()
