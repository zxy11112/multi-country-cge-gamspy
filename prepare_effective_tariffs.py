# -*- coding: utf-8 -*-
"""
Effective-tariff correction factors: anchor the model's benchmark tariff
revenue T0 to actual customs revenue (IMF GFS G1151) for 2022.

factor[importer] = G1151_usd / model_T0

The model's sector-level tariff STRUCTURE (from WITS HS6 MFN / indicators)
is kept; every rate into an importer is scaled by its factor. This fixes
the known distortions:
  - MFN-based rates overstate actual collections where FTAs/exemptions
    bite (CHN ~0.35, JPN ~0.62, KOR ~0.18);
  - the USA uniform AHS average misses 301/232 additional duties
    (USA factor ~2.6).
EU27 and ROW keep factor 1.0 (no complete G1151 coverage; documented
limitation).

Inputs : data_real_6x12/baseline_tariffs_2022.csv
         data_real_6x12/bilateral_flows_2022.csv
         GFS_SOO_<CC>_XDC.csv (IMF GFS, G1151 customs revenue, local ccy)
         World Bank PA.NUS.FCRF 2022 (official exchange rate)
Output : data_real_6x12/tariff_effective_factors.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\数据")
OUT = ROOT / "simple_multicountry_cge" / "data_real_6x12"
GFS_DIR = ROOT / "CGE公开数据" / "12_IMF_GFS" / "明细_按国家"

# 2022 official exchange rate, LCU per USD (World Bank PA.NUS.FCRF)
FX = {"CHN": 6.73715811237119, "JPN": 131.498140443764,
      "KOR": 1291.44666666667, "USA": 1.0}
CORRECTED = ["CHN", "USA", "JPN", "KOR"]

# EU27: customs duties are transferred to the EU budget, so national G1151
# is ~0 by construction. Use the EC figure instead: EUR 25 bn transferred
# in 2022 (net of the 25% collection-cost retention since 2021), i.e.
# gross collected ~= 25/0.75 = 33.3 bn EUR. EUR/USD 2022 avg = 1.0530.
# Sources: epthinktank.eu (2023-09-25), EC DG BUDG traditional own resources.
EU27_G1151_MUSD = 25_000 / 0.75 * 1.0530   # million USD


def g1151_usd(region: str) -> float:
    """Customs & import duties revenue, 2022, million USD."""
    df = pd.read_csv(GFS_DIR / f"GFS_SOO_{region}_XDC.csv")
    v = df[(df["SECTOR"] == "S13") & (df["INDICATOR"] == "G1151_T")
           & (df["TIME_PERIOD"] == 2022)]["OBS_VALUE"].iloc[0]
    # GFS XDC is in raw local currency units -> million USD
    return float(v) / FX[region] / 1e6


def main() -> None:
    tau = pd.read_csv(OUT / "baseline_tariffs_2022.csv")
    flows = pd.read_csv(OUT / "bilateral_flows_2022.csv")
    m = flows.merge(tau, on=["origin", "sector", "destination"], how="left")
    m["rate"] = m["rate"].fillna(0.0)
    t0 = (m["rate"] * m["value"]).groupby(m["destination"]).sum()  # mn USD

    rows = []
    for reg in ["CHN", "USA", "JPN", "KOR", "EU27", "ROW"]:
        model_t0 = float(t0.get(reg, 0.0))
        if reg in CORRECTED:
            actual = g1151_usd(reg)
            factor = actual / model_t0 if model_t0 > 0 else 1.0
        elif reg == "EU27":
            actual = EU27_G1151_MUSD
            factor = actual / model_t0 if model_t0 > 0 else 1.0
        else:
            actual = None
            factor = 1.0
        rows.append((reg, model_t0, actual, factor))
    out = pd.DataFrame(rows, columns=["region", "model_T0_mUSD",
                                      "g1151_mUSD", "factor"])
    out.to_csv(OUT / "tariff_effective_factors.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nwritten to {OUT / 'tariff_effective_factors.csv'}")


if __name__ == "__main__":
    main()
