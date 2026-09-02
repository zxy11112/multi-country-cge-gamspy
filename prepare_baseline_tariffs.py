# -*- coding: utf-8 -*-
"""
Build baseline (benchmark-year 2022) ad valorem tariff rates for the
CHN / USA / ROW x {prim, manu, serv} demo.

Output: data_real_io/baseline_tariffs_2022.csv
    origin, sector, destination, rate

Rate conventions (documented approximations):
- CHN imports (USA->CHN, ROW->CHN): China MFN 2022, HS6 simple average
  within each model sector (WITS HS6 file; MFN not AHS; specific duties
  not converted, so agricultural protection is understated).
- USA imports (CHN->USA, ROW->USA): USA AHS weighted-average 2022
  (WITS indicator, all products) applied uniformly to both goods
  sectors. Sector detail for 2022 is unavailable in the local data;
  the USITC snapshot is 2026 vintage and unsuitable for a 2022
  benchmark.
- ROW imports (CHN->ROW, USA->ROW): unweighted mean of AHS
  weighted-average 2022 across all WITS reporters except CHN/USA.
- ROW->ROW (intra-aggregate), domestic flows, services: 0.

Sector mapping: HS chapters 01-27 -> prim, 28-97 -> manu.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
CLEAN = ROOT / "整理" / "02_清洗"
OUT = ROOT / "simple_multicountry_cge" / "data_real_io"

HS6 = CLEAN / "clean_tariff_wits_hs6_2022_v1.csv"
IND = CLEAN / "clean_tariff_wits_indicators_v1.csv"

REGIONS = ["CHN", "USA", "ROW"]
SECTORS = ["prim", "manu", "serv"]


def chn_mfn_by_sector() -> dict[str, float]:
    df = pd.read_csv(HS6, usecols=["reporter_iso3", "hs_code",
                                   "duty_value", "duty_unit"],
                     dtype={"hs_code": str})
    df = df[(df["reporter_iso3"] == "CHN")
            & (df["duty_unit"] == "ad_valorem")]
    df["chapter"] = df["hs_code"].str[:2].astype(int)
    df["sector"] = np.where(df["chapter"] <= 27, "prim", "manu")
    g = df.groupby("sector")["duty_value"].mean() / 100.0
    print("CHN MFN 2022 simple average:",
          {k: f"{v:.4f}" for k, v in g.items()})
    return g.to_dict()


def indicator_avg(reporter: str | None, year: int = 2022) -> float:
    df = pd.read_csv(IND, usecols=["reporter_iso3", "year", "tariff_type",
                                   "duty_value"])
    df = df[(df["year"] == year) & (df["tariff_type"] == "AHS_WGHTD")]
    if reporter is not None:
        df = df[df["reporter_iso3"] == reporter]
    else:
        df = df[~df["reporter_iso3"].isin(["CHN", "USA"])]
    v = df["duty_value"].mean() / 100.0
    return float(v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chn = chn_mfn_by_sector()
    usa_ahs = indicator_avg("USA")
    row_ahs = indicator_avg(None)
    print(f"USA AHS_WGHTD 2022: {usa_ahs:.4f}")
    print(f"ROW mean AHS_WGHTD 2022: {row_ahs:.4f}")

    rows = []
    for o in REGIONS:
        for k in SECTORS:
            for s in REGIONS:
                rate = 0.0
                if o != s and k != "serv":
                    if s == "CHN":
                        rate = chn[k]
                    elif s == "USA":
                        rate = usa_ahs
                    else:  # s == "ROW"
                        rate = row_ahs
                rows.append((o, k, s, rate))
    df = pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                     "rate"])
    df.to_csv(OUT / "baseline_tariffs_2022.csv", index=False)
    print("\nbaseline tariffs:")
    print(df[df["rate"] > 0].round(4).to_string(index=False))
    print(f"\nwritten to {OUT / 'baseline_tariffs_2022.csv'}")


if __name__ == "__main__":
    main()
