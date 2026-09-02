# -*- coding: utf-8 -*-
"""
Extract 2022 bilateral services trade for the G20-flavor region set:
  CHN USA EU27 JPN KOR IND CAN MEX BRA ZAF RUS SAU AUS ROW
from BaTIS (balanced values, item S, export records).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
BATIS_DIR = ROOT / "CGE公开数据" / "15_WTO_OECD_BaTIS"
OUT = ROOT / "simple_multicountry_cge" / "data_real_g20"
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 2022
REGIONS = ["CHN", "USA", "EU27", "JPN", "KOR", "IND", "CAN", "MEX",
           "BRA", "ZAF", "RUS", "SAU", "AUS", "ROW"]
ISO2 = {"CHN": "CN", "USA": "US", "JPN": "JP", "KOR": "KR",
        "IND": "IN", "CAN": "CA", "MEX": "MX", "BRA": "BR",
        "ZAF": "ZA", "RUS": "RU", "SAU": "SA", "AUS": "AU"}
EU27_ISO2 = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
             "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
             "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
ITEM = "S"


def main() -> None:
    zip_path = next(BATIS_DIR.glob("OECD-WTO_BATIS_data*.zip"))
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [n for n in zf.namelist()
                    if n.lower().endswith(".csv")][0]
        cols = ["Reporter", "Partner", "Flow", "Item_code", "Year",
                "Balanced_value"]
        use = pd.read_csv(zf.open(csv_name), usecols=cols)
    use = use[(use["Year"] == YEAR) & (use["Item_code"] == ITEM)
              & (use["Flow"] == "X")]
    mat = use.pivot_table(index="Reporter", columns="Partner",
                          values="Balanced_value", aggfunc="sum").fillna(0.0)
    countries = [c for c in mat.index if c not in ("WL", "EU")]
    mat_c = mat.loc[countries]
    non_eu = [c for c in countries if c not in EU27_ISO2]

    def xv(rep, par) -> float:
        try:
            return float(mat.loc[rep, par])
        except KeyError:
            return 0.0

    def world_to(par) -> float:
        return float(mat_c[par].sum()) if par in mat_c.columns else 0.0

    nR = len(REGIONS)
    x = np.zeros((nR, nR))
    explicit = REGIONS[:-1]

    eu_x_to = {p: sum(xv(m, p) for m in EU27_ISO2)
               for p in list(ISO2.values()) + ["WL"]}
    eu_intra = sum(xv(m, m2) for m in EU27_ISO2 for m2 in EU27_ISO2)
    eu_extra = eu_x_to["WL"] - eu_intra
    eu_imp_extra = float(sum(xv(c, m) for c in non_eu for m in EU27_ISO2))

    eu_idx = REGIONS.index("EU27")
    for oi, o in enumerate(explicit):
        if o == "EU27":
            continue
        iso = ISO2[o]
        used = 0.0
        for sj, s_ in enumerate(explicit):
            if s_ == o:
                continue
            v = (sum(xv(iso, m) for m in EU27_ISO2) if s_ == "EU27"
                 else xv(iso, ISO2[s_]))
            x[oi, sj] = v
            used += v
        x[oi, -1] = max(xv(iso, "WL") - used, 0.0)

    for sj, s_ in enumerate(explicit):
        if s_ == "EU27":
            continue
        x[eu_idx, sj] = eu_x_to[ISO2[s_]]
    x[eu_idx, -1] = max(eu_extra - sum(x[eu_idx, sj]
                                       for sj in range(nR - 1)), 0.0)

    for sj, s_ in enumerate(explicit):
        world_s = eu_imp_extra if s_ == "EU27" else world_to(ISO2[s_])
        x[-1, sj] = max(world_s - x[:-1, sj].sum(), 0.0)
    wl_total = world_to("WL")
    x[-1, -1] = max(wl_total - x[:-1, :].sum() - x[-1, :-1].sum(), 0.0)

    print("services flows x0[SRV] (bn USD):")
    print((pd.DataFrame(x, index=REGIONS, columns=REGIONS) / 1e3)
          .round(1).to_string())
    rows = [(REGIONS[o], "SRV", REGIONS[s], float(x[o, s]))
            for o in range(nR) for s in range(nR)]
    pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                "value"]).to_csv(
        OUT / "services_flows_batis_2022.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
