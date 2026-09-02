# -*- coding: utf-8 -*-
"""
Extract 2022 bilateral services trade for the 6-region model from the
OECD-WTO BaTIS dataset (BPM6 edition).

Uses BALANCED values (reconciled flows), item S (total services),
export records, ISO2 -> model-region mapping:
  CN->CHN, US->USA, JP->JPN, KR->KOR, EU->EU27.
ROW flows are computed as residuals from world (WL) totals:
  x[ROW -> s]   = x[WL -> s] - sum(explicit origins -> s)
  x[o   -> ROW] = x[o -> WL] - sum(o -> other explicit regions)

Output (million USD):
  data_real_6x12/services_flows_batis_2022.csv
      origin, sector(=SRV), destination, value
  data_real_6x12/services_shares_batis_2022.csv
      origin, destination, share   (diagnostics: origin shares per dest)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
BATIS_DIR = ROOT / "CGE公开数据" / "15_WTO_OECD_BaTIS"
OUT = ROOT / "simple_multicountry_cge" / "data_real_6x12"

YEAR = 2022
REGIONS = ["CHN", "USA", "JPN", "KOR", "EU27", "ROW"]
ISO2 = {"CHN": "CN", "USA": "US", "JPN": "JP", "KOR": "KR"}
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
    print(f"rows for {YEAR}/{ITEM}/X: {len(use)}")

    mat = use.pivot_table(index="Reporter", columns="Partner",
                          values="Balanced_value", aggfunc="sum") \
        .fillna(0.0)
    # drop aggregate reporters (WL, EU) to avoid double counting
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

    # EU27 aggregates (BaTIS has no EU-partner detail; aggregate members)
    eu_x_to = {p: sum(xv(m, p) for m in EU27_ISO2)
               for p in ["CN", "US", "JP", "KR", "WL"]}
    eu_intra = sum(xv(m, m2) for m in EU27_ISO2 for m2 in EU27_ISO2)
    eu_extra = eu_x_to["WL"] - eu_intra
    print(f"EU27 extra-EU services exports: {eu_extra:,.0f} mn USD")
    # extra-EU imports: non-EU country reporters -> EU27 members
    eu_imp_extra = float(sum(xv(c, m) for c in non_eu for m in EU27_ISO2))
    print(f"EU27 extra-EU services imports: {eu_imp_extra:,.0f} mn USD")

    # ---------- build 6-region flows ----------
    x = np.zeros((6, 6))
    # explicit origins (non-EU) -> destinations
    for oi, o in enumerate(["CHN", "USA", "JPN", "KOR"]):
        iso = ISO2[o]
        vals = {}
        for sj, s_ in enumerate(["CHN", "USA", "JPN", "KOR"]):
            if s_ == o:
                continue
            x[oi, sj] = xv(iso, ISO2[s_])
            vals[s_] = x[oi, sj]
        x[oi, 4] = sum(xv(iso, m) for m in EU27_ISO2)   # -> EU27
        used = sum(vals.values()) + x[oi, 4]
        x[oi, -1] = max(xv(iso, "WL") - used, 0.0)       # -> ROW
    # EU27 origin
    x[4, 0] = eu_x_to["CN"]
    x[4, 1] = eu_x_to["US"]
    x[4, 2] = eu_x_to["JP"]
    x[4, 3] = eu_x_to["KR"]
    x[4, -1] = max(eu_extra - (x[4, 0] + x[4, 1] + x[4, 2] + x[4, 3]), 0.0)
    # ROW origin -> explicit destinations (world totals minus explicit)
    x[-1, 0] = max(world_to("CN") - x[0:5, 0].sum(), 0.0)
    x[-1, 1] = max(world_to("US") - x[0:5, 1].sum(), 0.0)
    x[-1, 2] = max(world_to("JP") - x[0:5, 2].sum(), 0.0)
    x[-1, 3] = max(world_to("KR") - x[0:5, 3].sum(), 0.0)
    x[-1, 4] = max(eu_imp_extra - x[0:5, 4].sum(), 0.0)
    # ROW -> ROW: residual so the matrix is consistent with world total
    wl_total = world_to("WL")
    x[-1, -1] = max(wl_total - x[:5, :].sum() - x[5, :5].sum(), 0.0)

    print("\nservices flows x0[SRV] (bn USD):")
    df = pd.DataFrame(x, index=REGIONS, columns=REGIONS) / 1e3
    print(df.round(1).to_string())

    # ---------- write ----------
    rows = [(REGIONS[o], "SRV", REGIONS[s], float(x[o, s]))
            for o in range(6) for s in range(6)]
    flows = pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                        "value"])
    flows.to_csv(OUT / "services_flows_batis_2022.csv", index=False)

    sh = pd.DataFrame(
        [(REGIONS[o], REGIONS[s],
          float(x[o, s] / x[:, s].sum()) if x[:, s].sum() > 0 else 0.0)
         for o in range(6) for s in range(6) if o != s],
        columns=["origin", "destination", "share"])
    sh.to_csv(OUT / "services_shares_batis_2022.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
