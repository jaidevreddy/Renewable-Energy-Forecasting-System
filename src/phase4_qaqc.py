import os
import pandas as pd

PH3_PATH   = "data/processed/daily_energy.parquet"
OUT_DETAIL = "data/processed/phase4_qaqc.parquet"
OUT_ZONE   = "data/processed/phase4_qaqc_summary.csv"

SYSTEM_KW = 10.0  # DC system size used in Phase 3

# QA thresholds (full-year)
TH_FULL = dict(
    min_days=330,         # require near-complete coverage
    max_zero_days=40,     # too many zero-energy days indicates issues
    cap_factor_lo=0.12,   # 12% (reasonable for BLR with fixed-tilt 10 kW)
    cap_factor_hi=0.28,   # 28% upper sanity
    mean_kwh_lo=5,        # daily mean sanity check
    mean_kwh_hi=60,
)
FULL_YEAR_DAY_MIN = 360   # classify "full year" vs "partial" for QA decision

def main():
    os.makedirs("data/processed", exist_ok=True)

    # ---------- Load phase-3 daily energy ----------
    de = pd.read_parquet(PH3_PATH)  # expects columns: ZoneID, date, energy_kwh
    de["date"] = pd.to_datetime(de["date"])
    de["year"] = de["date"].dt.year

    # ---------- Per zone-year aggregates ----------
    def agg(g):
        days = g["date"].nunique()
        zero_days = (g["energy_kwh"] <= 0.01).sum()
        annual_kwh = g["energy_kwh"].sum()
        cap_factor = annual_kwh / (SYSTEM_KW * 24 * 365)
        return pd.Series(dict(
            days=days,
            zero_days=zero_days,
            annual_kwh=annual_kwh,
            cap_factor=cap_factor,
            mean_kwh=g["energy_kwh"].mean(),
            p5=g["energy_kwh"].quantile(0.05),
            p95=g["energy_kwh"].quantile(0.95),
        ))

    stats = de.groupby(["ZoneID", "year"], as_index=False).apply(agg)

    # ---------- Rule booleans ----------
    s = stats  # alias
    s["is_full_year"] = s["days"] >= FULL_YEAR_DAY_MIN

    s["ok_days"]      = s["days"] >= TH_FULL["min_days"]
    s["ok_zero_days"] = s["zero_days"] <= TH_FULL["max_zero_days"]
    s["ok_cf"]        = s["cap_factor"].between(TH_FULL["cap_factor_lo"], TH_FULL["cap_factor_hi"])
    s["ok_mean"]      = s["mean_kwh"].between(TH_FULL["mean_kwh_lo"], TH_FULL["mean_kwh_hi"])

    # QA pass for a given zone-year (only meaningful for full years)
    s["qa_pass_year"] = s[["ok_days", "ok_zero_days", "ok_cf", "ok_mean"]].all(axis=1)

    # ---------- Zone-level pass/fail ----------
    # Consider ONLY full years for the zone pass decision.
    # Partial years are retained in the details parquet but do not count against the zone.
    full = s[s["is_full_year"]].copy()

    zone_summary = (
        s.groupby("ZoneID", as_index=False)
         .agg(n_years=("year", "nunique"),
              n_full_years=("is_full_year", "sum"))
         .merge(
             full.groupby("ZoneID", as_index=False)
                 .agg(n_full_pass=("qa_pass_year", "sum")),
             on="ZoneID", how="left"
         )
    )
    zone_summary["n_full_pass"] = zone_summary["n_full_pass"].fillna(0).astype(int)
    zone_summary["qa_pass_zone"] = zone_summary["n_full_pass"] > 0

    # Add the most recent full year evaluated
    recent_full = (full.sort_values(["ZoneID", "year"])
                        .groupby("ZoneID").tail(1)[["ZoneID", "year", "qa_pass_year"]]
                        .rename(columns={"year": "latest_full_year",
                                         "qa_pass_year": "latest_full_year_pass"}))
    zone_summary = zone_summary.merge(recent_full, on="ZoneID", how="left")

    # ---------- Write outputs ----------
    s.sort_values(["ZoneID", "year"], inplace=True)
    s.to_parquet(OUT_DETAIL, index=False)
    zone_summary.sort_values("ZoneID").to_csv(OUT_ZONE, index=False)

    # ---------- Console summary ----------
    n_zones = zone_summary["ZoneID"].nunique()
    n_pass  = int(zone_summary["qa_pass_zone"].sum())
    n_full0 = int((zone_summary["n_full_years"] == 0).sum())

    print(f"[phase4] wrote {OUT_DETAIL} (per zone–year) and {OUT_ZONE} (zone summary).")
    print(f"[phase4] zones passing QA: {n_pass} / {n_zones}")
    if n_full0:
        print(f"[phase4][note] {n_full0} zone(s) have no full years (all partial) — "
              "they neither pass nor fail; check details parquet.")

if __name__ == "__main__":
    main()
