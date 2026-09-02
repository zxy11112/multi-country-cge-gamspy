# -*- coding: utf-8 -*-
"""
Baseline (2022) ad valorem tariffs for the 6-region x 12-sector model.

Regions: CHN, USA, JPN, KOR, EU27, ROW
Sectors: AGF MIN ENR CHM TXL WDP NMM MAC ELE VEH OTM SRV

Rates:
- CHN/JPN/KOR imports: own MFN 2022, HS6 simple average per sector
  (WITS HS6 files, chapters mapped to model sectors).
- EU27 imports: DEU file (= EU common external tariff), same method.
- USA imports: USA AHS weighted-average 2022 (WITS indicator), uniform.
- ROW imports: unweighted mean of AHS weighted-average 2022 across
  reporters excluding CHN/USA/JPN/KOR and EU27 members.
- ROW->ROW, domestic, SRV: 0.

Output: data_real_6x12/baseline_tariffs_2022.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
CLEAN = ROOT / "整理" / "02_清洗"
OUT = ROOT / "simple_multicountry_cge" / "data_real_6x12"

HS6 = CLEAN / "clean_tariff_wits_hs6_2022_v1.csv"
IND = CLEAN / "clean_tariff_wits_indicators_v1.csv"

REGIONS = ["CHN", "USA", "JPN", "KOR", "EU27", "ROW"]
SECTORS = ["AGF", "MIN", "ENR", "CHM", "TXL", "WDP",
           "NMM", "MAC", "ELE", "VEH", "OTM", "SRV"]
EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
        "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
        "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]


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


def mfn_by_sector(reporter: str) -> dict[str, float]:
    df = pd.read_csv(HS6, usecols=["reporter_iso3", "hs_code",
                                   "duty_value", "duty_unit"],
                     dtype={"hs_code": str})
    df = df[(df["reporter_iso3"] == reporter)
            & (df["duty_unit"] == "ad_valorem")]
    df["sector"] = df["hs_code"].str[:2].astype(int).map(hs_sector)
    g = df.groupby("sector")["duty_value"].mean() / 100.0
    return g.reindex(SECTORS[:-1]).to_dict()


def indicator_avg(exclude: list[str] | None, reporter: str | None) -> float:
    df = pd.read_csv(IND, usecols=["reporter_iso3", "year", "tariff_type",
                                   "duty_value"])
    df = df[(df["year"] == 2022) & (df["tariff_type"] == "AHS_WGHTD")]
    if reporter is not None:
        df = df[df["reporter_iso3"] == reporter]
    elif exclude is not None:
        df = df[~df["reporter_iso3"].isin(exclude)]
    return float(df["duty_value"].mean() / 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rates = {
        "CHN": mfn_by_sector("CHN"),
        "JPN": mfn_by_sector("JPN"),
        "KOR": mfn_by_sector("KOR"),
        "EU27": mfn_by_sector("DEU"),          # EU common external tariff
    }
    usa_ahs = indicator_avg(None, "USA")
    row_ahs = indicator_avg(EU27 + ["CHN", "USA", "JPN", "KOR"], None)
    print("MFN simple averages by sector:")
    print(pd.DataFrame(rates).round(4).to_string())
    print(f"USA AHS {usa_ahs:.4f} | ROW mean AHS {row_ahs:.4f}")

    rows = []
    for o in REGIONS:
        for k in SECTORS:
            for s in REGIONS:
                rate = 0.0
                if o != s and k != "SRV":
                    if s in rates:
                        rate = rates[s][k]
                    elif s == "USA":
                        rate = usa_ahs
                    else:
                        rate = row_ahs
                rows.append((o, k, s, rate))
    df = pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                     "rate"])
    df.to_csv(OUT / "baseline_tariffs_2022.csv", index=False)
    print(f"\nnonzero entries: {(df['rate'] > 0).sum()} / {len(df)}")
    print(f"written to {OUT / 'baseline_tariffs_2022.csv'}")


if __name__ == "__main__":
    main()
