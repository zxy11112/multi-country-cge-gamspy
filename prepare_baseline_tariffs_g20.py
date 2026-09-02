# -*- coding: utf-8 -*-
"""
Baseline (2022) ad valorem tariffs for the 14-region x 12-sector model.

Regions: CHN USA JPN KOR EU27 VNM THA MYS IDN SGP PHL IND MEX ROW

Rates:
- CHN/JPN/KOR/EU27(DEU): WITS HS6 2022 MFN simple average per sector
  (existing clean_tariff_wits_hs6_2022_v1.csv).
- 8 expansion countries: WITS HS6 2022 MFN from
  clean_tariff_wits_hs6_2022_expansion.csv (download_wits_expansion.py).
- USA: AHS_WGHTD 2022 (WITS indicator), uniform.
- ROW: mean AHS_WGHTD 2022 over reporters excluding the 13 explicit ones.

Output: data_real_g20/baseline_tariffs_2022.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
CLEAN = ROOT / "整理" / "02_清洗"
OUT = ROOT / "simple_multicountry_cge" / "data_real_g20"

REGIONS = ["CHN", "USA", "EU27", "JPN", "KOR", "IND", "CAN", "MEX",
           "BRA", "ZAF", "RUS", "SAU", "AUS", "ROW"]
SECTORS = ["AGF", "MIN", "ENR", "CHM", "TXL", "WDP",
           "NMM", "MAC", "ELE", "VEH", "OTM", "SRV"]
EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
        "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
        "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
NEW6 = ["IND", "CAN", "MEX", "BRA", "ZAF", "RUS", "SAU", "AUS"]


def hs_sector(ch: int) -> str:
    if ch <= 24:
        return "AGF"
    if ch <= 26:
        return "MIN"
    if ch == 27:
        return "ENR"
    if ch <= 40:
        return "CHM"
    if ch <= 43:
        return "TXL"
    if ch <= 49:
        return "WDP"
    if ch <= 67:
        return "TXL"
    if ch <= 83:
        return "NMM"
    if ch == 84:
        return "MAC"
    if ch == 85:
        return "ELE"
    if ch <= 89:
        return "VEH"
    return "OTM"


def mfn_by_sector(df: pd.DataFrame, iso: str) -> dict[str, float]:
    d = df[(df["reporter_iso3"] == iso) & (df["duty_unit"] == "ad_valorem")]
    d = d.copy()
    d["sector"] = d["hs_code"].astype(str).str[:2].astype(int).map(hs_sector)
    d["duty_value"] = pd.to_numeric(d["duty_value"], errors="coerce")
    g = d.groupby("sector")["duty_value"].mean() / 100.0
    return g.reindex(SECTORS[:-1]).to_dict()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(CLEAN / "clean_tariff_wits_hs6_2022_v1.csv",
                       dtype={"hs_code": str})
    # expansion HS6: use only files that parse completely; others fall back
    # to the country-level AHS indicator (uniform rate) — documented.
    exp_file = CLEAN / "clean_tariff_wits_hs6_2022_expansion.csv"
    exp = pd.read_csv(exp_file, dtype={"hs_code": str}) \
        if exp_file.exists() else pd.DataFrame()
    if len(exp):
        exp = exp.copy()
        exp["duty_value"] = pd.to_numeric(exp["duty_value"], errors="coerce")

    rates = {"CHN": mfn_by_sector(base, "CHN"),
             "JPN": mfn_by_sector(base, "JPN"),
             "KOR": mfn_by_sector(base, "KOR"),
             "EU27": mfn_by_sector(base, "DEU")}
    for iso in NEW6:
        sub = exp[exp["reporter_iso3"] == iso] if len(exp) else exp
        if len(sub) > 4000:      # near-complete HS6 coverage (~5,400 lines)
            rates[iso] = mfn_by_sector(exp, iso)
            print(f"{iso}: HS6 detail ({len(sub):,} lines)")
        else:
            print(f"{iso}: fallback to AHS indicator (uniform)")

    ind = pd.read_csv(CLEAN / "clean_tariff_wits_indicators_v1.csv")
    ahs = ind[ind["tariff_type"] == "AHS_WGHTD"].copy()
    ahs = ahs.sort_values("year").groupby("reporter_iso3").tail(1)  # latest year per country
    ahs_by_country = (ahs.groupby("reporter_iso3")["duty_value"]
                      .mean() / 100).to_dict()
    usa_ahs = float(ahs_by_country.get("USA", 0.0156))
    row_ahs = float(ahs[~ahs["reporter_iso3"].isin(
        EU27 + ["CHN", "USA", "JPN", "KOR"] + NEW6)]["duty_value"].mean()) / 100
    print(f"USA AHS {usa_ahs:.4f} | ROW mean AHS {row_ahs:.4f}")
    print(pd.DataFrame(rates).round(4).to_string())

    rows = []
    for o in REGIONS:
        for k in SECTORS:
            for s in REGIONS:
                rate = 0.0
                if o != s and k != "SRV":
                    if s in rates:
                        rate = rates[s].get(k) or 0.0
                    elif s in ahs_by_country:
                        rate = float(ahs_by_country[s])  # uniform fallback
                    else:
                        rate = row_ahs
                rows.append((o, k, s, rate))
    df = pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                     "rate"])
    df.to_csv(OUT / "baseline_tariffs_2022.csv", index=False)
    print(f"\nwritten to {OUT / 'baseline_tariffs_2022.csv'}")


if __name__ == "__main__":
    main()
