# -*- coding: utf-8 -*-
"""
Build a real-data benchmark for the 3-region (CHN / USA / ROW) x 2-sector
(prim / manu) CGE demo, benchmark year 2022.

Sources
-------
1. OECD harmonised national IOTs, VA components (2025 release):
   CGE公开数据/06_OECD_IOT/IOTs_VA.zip  ->  VAcomponents.csv
   Variables: VALU (value added), LABR (compensation of employees);
   capital is computed as VALU - LABR (= CFC + NOPS + OTXS).
   ROW = sum over all economies except CHN and USA.
2. Cleaned UN Comtrade SITC long table:
   整理/02_清洗/clean_comtrade_sitc_1992-2023_v1.1.csv
   Reporters CHN/USA export records (X, FOB) for their own exports;
   ROW exports to CHN/USA use the importers' M records (CIF mirror).

Sector mapping (demo-grade approximation, SITC<->ISIC Rev.4):
   prim : SITC 1-digit 0-4   <-> ISIC A01-A03 + B05-B09
   manu : SITC 1-digit 5-9   <-> ISIC C* (all manufacturing divisions)
SITC 0-1 (food) partly overlaps ISIC C10T12 (food processing); accepted
for this demo and documented in the output metadata.

Known limitations (demo only):
   * Output Y0 is value added, while trade flows are gross values.
     The no-intermediate-input model requires Y0 = factor payments,
     so flows are interpreted as consistent accounting entries, not
     literal gross-output shipments.
   * CHN/USA imports use CIF mirror values; exports use FOB.
   * Baseline tariffs are set to zero; the counterfactual applies a
     fresh tariff on top.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
VA_ZIP = ROOT / "CGE公开数据" / "06_OECD_IOT" / "IOTs_VA.zip"
COMTRADE = ROOT / "整理" / "02_清洗" / "clean_comtrade_sitc_1992-2023_v1.1.csv"
OUT = ROOT / "simple_multicountry_cge" / "data_real"

YEAR = 2022
REGIONS = ["CHN", "USA", "ROW"]
SECTORS = ["prim", "manu"]

PRIM_ISIC = ["A01", "A02", "A03", "B05", "B06", "B07", "B08", "B09"]


def load_va() -> pd.DataFrame:
    """VA by country x model sector for YEAR, million USD."""
    with zipfile.ZipFile(VA_ZIP) as zf:
        with zf.open("VAcomponents.csv") as fh:
            va = pd.read_csv(fh)
    va = va[va["year"] == YEAR]
    va["sector"] = np.where(va["industry"].isin(PRIM_ISIC), "prim",
                            np.where(va["industry"].str.startswith("C"), "manu",
                                     None))
    va = va.dropna(subset=["sector"])
    wide = va.pivot_table(index=["country", "sector"], columns="variable",
                          values="mlln_USD", aggfunc="sum").reset_index()
    wide["capital"] = wide["VALU"] - wide["LABR"]
    wide = wide.rename(columns={"VALU": "Y0", "LABR": "labor"})
    return wide[["country", "sector", "Y0", "labor", "capital"]]


def load_trade() -> pd.DataFrame:
    """Bilateral goods trade among {CHN, USA, WLD}, YEAR, SITC 1-digit."""
    cols = ["reporter_iso3", "partner_iso3", "flow_code", "cmd_code",
            "period", "cl_code", "primary_value"]
    use = pd.read_csv(
        COMTRADE, usecols=cols,
        dtype={"reporter_iso3": str, "partner_iso3": str,
               "flow_code": str, "cmd_code": str},
    )
    use = use[(use["period"] == YEAR) & (use["cl_code"] == "S4")
              & (use["cmd_code"] != "TOTAL")
              & (use["reporter_iso3"].isin(["CHN", "USA"]))]
    use = use[use["cmd_code"].str.len() == 2]              # SITC 2-digit
    use["sector"] = np.where(use["cmd_code"].str[0].astype(int) <= 4,
                             "prim", "manu")
    keep = use["partner_iso3"].isin(["CHN", "USA", "WLD"])
    use = use[keep & use["flow_code"].isin(["X", "M"])]
    g = (use.groupby(["reporter_iso3", "partner_iso3", "flow_code", "sector"])
            ["primary_value"].sum().reset_index())
    g["mlln_USD"] = g["primary_value"] / 1.0e6
    return g


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- production side ----------
    va = load_va()
    va3 = (
        va[va["country"].isin(["CHN", "USA"])]
        .set_index(["country", "sector"])
    )
    row = (
        va[~va["country"].isin(["CHN", "USA"])]
        .groupby("sector")[["Y0", "labor", "capital"]].sum()
    )
    records = []
    for reg in ["CHN", "USA"]:
        for sec in SECTORS:
            v = va3.loc[(reg, sec)]
            records.append((reg, sec, v["Y0"], v["labor"], v["capital"]))
    for sec in SECTORS:
        v = row.loc[sec]
        records.append(("ROW", sec, v["Y0"], v["labor"], v["capital"]))
    prod_df = pd.DataFrame(
        records, columns=["region", "sector", "Y0", "labor", "capital"])

    # ---------- trade side ----------
    tr = load_trade().set_index(
        ["reporter_iso3", "partner_iso3", "flow_code", "sector"])["mlln_USD"]

    def x(rep, par, sec, flow="X") -> float:
        try:
            return float(tr[(rep, par, flow, sec)])
        except KeyError:
            return 0.0

    x0 = np.zeros((3, 2, 3))          # [origin, sector, destination]
    for k, sec in enumerate(SECTORS):
        chn_x_wld = x("CHN", "WLD", sec)
        usa_x_wld = x("USA", "WLD", sec)
        chn_m_wld = x("CHN", "WLD", sec, "M")
        usa_m_wld = x("USA", "WLD", sec, "M")

        x_chn_usa = x("CHN", "USA", sec)          # CHN exports -> USA (FOB)
        x_usa_chn = x("USA", "CHN", sec)          # USA exports -> CHN (FOB)
        x_chn_row = chn_x_wld - x_chn_usa         # CHN exports -> ROW
        x_usa_row = usa_x_wld - x_usa_chn         # USA exports -> ROW
        x_row_chn = chn_m_wld - x_usa_chn         # ROW -> CHN (CIF mirror)
        x_row_usa = usa_m_wld - x_chn_usa         # ROW -> USA (CIF mirror)

        # origin index: CHN=0, USA=1, ROW=2 ; destination likewise
        x0[0, k, 1] = x_chn_usa
        x0[0, k, 2] = x_chn_row
        x0[1, k, 0] = x_usa_chn
        x0[1, k, 2] = x_usa_row
        x0[2, k, 0] = x_row_chn
        x0[2, k, 1] = x_row_usa

    # ---------- domestic flows as residuals (guarantee Y0 = sum_s x0) ----------
    Y0 = prod_df.set_index(["region", "sector"])["Y0"]
    resid_report = []
    for oi, reg in enumerate(REGIONS):
        for k, sec in enumerate(SECTORS):
            dom = Y0[(reg, sec)] - x0[oi, k, :].sum() + x0[oi, k, oi]
            # x0[oi,k,oi] is currently 0; domestic = Y0 - exports
            dom = Y0[(reg, sec)] - (x0[oi, k, :].sum())
            x0[oi, k, oi] = dom
            resid_report.append((reg, sec, Y0[(reg, sec)], dom,
                                 dom / Y0[(reg, sec)]))
    rep = pd.DataFrame(resid_report,
                       columns=["region", "sector", "Y0", "domestic_flow",
                                "domestic_share"])

    # ---------- diagnostics ----------
    print("=== production (million USD, 2022) ===")
    print(prod_df.round(0).to_string(index=False))
    print("\n=== domestic flow residuals (must be > 0) ===")
    print(rep.round(3).to_string(index=False))
    assert (rep["domestic_flow"] > 0).all(), "negative domestic flow!"

    flows = pd.DataFrame(
        [(REGIONS[o], SECTORS[k], REGIONS[s], x0[o, k, s])
         for o in range(3) for k in range(2) for s in range(3)],
        columns=["origin", "sector", "destination", "value"])
    print("\n=== bilateral flows x0 (million USD) ===")
    print(flows.round(0).to_string(index=False))
    assert (flows["value"] > 0).all()

    # trade balances -> fixed net foreign transfers B (sum is 0 by construction)
    B = {}
    for si, reg in enumerate(REGIONS):
        imports = sum(x0[oi, k, si] for oi in range(3) if oi != si
                      for k in range(2))
        exports = sum(x0[si, k, dj] for dj in range(3) if dj != si
                      for k in range(2))
        B[reg] = imports - exports
    print("\n=== net foreign transfers B = M - X (million USD) ===")
    for reg in REGIONS:
        print(f"  {reg}: {B[reg]:,.0f}")
    print(f"  sum: {sum(B.values()):.6f}")

    # ---------- write ----------
    prod_df.to_csv(OUT / "production_va_2022.csv", index=False)
    flows.to_csv(OUT / "bilateral_flows_2022.csv", index=False)
    pd.DataFrame({"region": REGIONS,
                  "net_foreign_transfer": [B[r] for r in REGIONS]}
                 ).to_csv(OUT / "net_transfers_2022.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    build()
