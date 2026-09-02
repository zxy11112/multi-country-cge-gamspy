# -*- coding: utf-8 -*-
"""
Build a real-data benchmark WITH intermediate inputs for the
3-region (CHN / USA / ROW) x 3-sector (prim / manu / serv) CGE demo,
benchmark year 2022.

Sources
-------
1. OECD harmonised national IOTs, 2025 release:
   - IOTs_DOMIMP.zip -> <CC><year>dom.csv : per-country tables with
     DOM_<ind> rows (domestic products), IMP_<ind> rows (imports),
     TXS_INT_FNL, VALU, OUTPUT rows; columns = 50 industries plus
     HFCE, NPISH, GGFC, GFCF, INVNT, DPABR, CONS_NONRES, EXPO, IMPO, TOTAL.
   - IOTs_VA.zip -> VAcomponents.csv : LABR (compensation of employees).
   ROW = sum over all economies except CHN and USA.
2. Cleaned UN Comtrade SITC long table (reporters CHN/USA only):
   used to split each importer's imports by origin (CHN / USA / ROW).

Sector mapping (ISIC Rev.4 -> model sector):
   prim : A01-A03, B05-B09      (agriculture + mining)
   manu : C10T12 .. C31T33      (all manufacturing)
   serv : D .. T                (all services, incl. utilities/construction)

Services bilateral trade is NOT in Comtrade; origin shares of services
imports are proxied by each importer's GOODS origin shares (documented
approximation).

Accounting conventions
----------------------
- x0[origin, sector, destination]: total bilateral USE (intermediate +
  final) of each origin-sector variety. Domestic flows = domestic products
  used at home (DOM rows minus EXPO/CONS_NONRES columns).
- io0[input, region, output]: composite (DOM+IMP) intermediate coefficient
  per unit of gross output.
- Factor payments: labor = LABR; capital = residual
  (OUTPUT - intermediates - LABR), i.e. operating surplus + CFC + net
  product taxes. This enforces exact column balance.
- Final demand C0: composite rows x (HFCE+NPISH+GGFC+GFCF+INVNT+DPABR).
  CONS_NONRES (non-residents' direct purchases) counts as exports.
- Net foreign transfer B = imports - exports at producer prices;
  household income I0 = sum of C0; the IOT identity M - X = C0 - VA
  makes the closure consistent.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
IOT_DIR = ROOT / "CGE公开数据" / "06_OECD_IOT"
DOMIMP_ZIP = IOT_DIR / "IOTs_DOMIMP.zip"
VA_ZIP = IOT_DIR / "IOTs_VA.zip"
COMTRADE = ROOT / "整理" / "02_清洗" / "clean_comtrade_sitc_1992-2023_v1.1.csv"
OUT = ROOT / "simple_multicountry_cge" / "data_real_io"

YEAR = 2022
REGIONS = ["CHN", "USA", "ROW"]
SECTORS = ["prim", "manu", "serv"]

PRIM = ["A01", "A02", "A03", "B05", "B06", "B07", "B08", "B09"]
# final demand split: household consumption / government / investment
FD_C = ["HFCE", "NPISH", "DPABR"]
FD_G = ["GGFC"]
FD_I = ["GFCF", "INVNT"]
FD_COLS = FD_C + FD_G + FD_I

_cache: dict[str, pd.DataFrame] = {}


def sector_of(ind: str) -> str:
    if ind in PRIM:
        return "prim"
    if ind.startswith("C"):
        return "manu"
    return "serv"


def load_dom(country: str) -> pd.DataFrame | None:
    """Aggregate one country's DOM/IMP table to 3 model sectors."""
    name = f"{country}{YEAR}dom.csv"
    with zipfile.ZipFile(DOMIMP_ZIP) as zf:
        if name not in zf.namelist():
            return None
        with zf.open(name) as fh:
            df = pd.read_csv(fh, index_col=0)
    df.index = df.index.str.strip('"')
    df.columns = df.columns.str.strip('"')

    ind_cols = [c for c in df.columns if c not in
                FD_COLS + ["CONS_NONRES", "EXPO", "IMPO", "TOTAL"]]
    sec_cols = {s: [c for c in ind_cols if sector_of(c) == s] for s in SECTORS}

    def block(prefix: str) -> pd.DataFrame:
        rows = df.loc[[i for i in df.index if i.startswith(prefix)]]
        rows = rows.copy()
        rows["sec"] = [sector_of(i[len(prefix):]) for i in rows.index]
        return rows.groupby("sec").sum()

    out = {}
    for pfx, key in [("DOM_", "dom"), ("IMP_", "imp")]:
        blk = block(pfx)
        out[key + "_int"] = pd.DataFrame(
            {s: blk[sec_cols[s]].sum(axis=1) for s in SECTORS})  # [input, output]
        out[key + "_fd_c"] = blk[FD_C].sum(axis=1)   # household consumption
        out[key + "_fd_g"] = blk[FD_G].sum(axis=1)   # government consumption
        out[key + "_fd_i"] = blk[FD_I].sum(axis=1)   # investment (GFCF+INVNT)
        out[key + "_fd"] = blk[FD_COLS].sum(axis=1)  # total final demand
        out[key + "_expo"] = blk["EXPO"] + blk["CONS_NONRES"]
        out[key + "_total"] = blk["TOTAL"]

    # composite (DOM + IMP)
    out["ttl_int"] = out["dom_int"] + out["imp_int"]
    out["ttl_fd"] = out["dom_fd"] + out["imp_fd"]
    out["ttl_fd_c"] = out["dom_fd_c"] + out["imp_fd_c"]
    out["ttl_fd_g"] = out["dom_fd_g"] + out["imp_fd_g"]
    out["ttl_fd_i"] = out["dom_fd_i"] + out["imp_fd_i"]
    # NOTE: the TXS_INT_FNL row is NOT added to final demand. The IOT row
    # identity is  FD_products = VA_eff + (M - X)  with VA_eff = OUTPUT minus
    # product-row intermediates; product taxes on intermediates sit inside
    # VA_eff (the capital residual), so C0 must be product rows only.

    out["output"] = df.loc["OUTPUT", ind_cols].groupby(
        [sector_of(c) for c in ind_cols]).sum()
    out["valu"] = df.loc["VALU", ind_cols].groupby(
        [sector_of(c) for c in ind_cols]).sum()
    return out


def get(country: str) -> dict:
    if country not in _cache:
        _cache[country] = load_dom(country)
    return _cache[country]


def load_labr() -> pd.Series:
    with zipfile.ZipFile(VA_ZIP) as zf:
        with zf.open("VAcomponents.csv") as fh:
            va = pd.read_csv(fh)
    va = va[(va["year"] == YEAR) & (va["variable"] == "LABR")]
    va["sec"] = va["industry"].map(sector_of)
    return va.groupby(["country", "sec"])["mlln_USD"].sum()


def load_trade_shares() -> pd.DataFrame:
    """Goods trade among CHN/USA/WLD by model sector + all-goods total."""
    cols = ["reporter_iso3", "partner_iso3", "flow_code", "cmd_code",
            "period", "cl_code", "primary_value"]
    use = pd.read_csv(
        COMTRADE, usecols=cols,
        dtype={"reporter_iso3": str, "partner_iso3": str,
               "flow_code": str, "cmd_code": str})
    use = use[(use["period"] == YEAR) & (use["cl_code"] == "S4")
              & (use["cmd_code"] != "TOTAL")
              & (use["reporter_iso3"].isin(["CHN", "USA"]))]
    use = use[use["cmd_code"].str.len() == 2]
    use["sec"] = np.where(use["cmd_code"].str[0].astype(int) <= 4,
                          "prim", "manu")
    use = use[use["partner_iso3"].isin(["CHN", "USA", "WLD"])
              & use["flow_code"].isin(["X", "M"])]
    g = (use.groupby(["reporter_iso3", "partner_iso3", "flow_code", "sec"])
            ["primary_value"].sum() / 1e6)
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labr = load_labr()
    tr = load_trade_shares()

    def t(rep, par, flow, sec):
        try:
            return float(tr[(rep, par, flow, sec)])
        except KeyError:
            return 0.0

    # ---------- per-region aggregates ----------
    agg = {}
    for reg in REGIONS:
        if reg == "ROW":
            with zipfile.ZipFile(DOMIMP_ZIP) as zf:
                countries = sorted({n[:3] for n in zf.namelist()
                                    if n.endswith(f"{YEAR}dom.csv")})
            row_parts = [get(c) for c in countries if c not in ("CHN", "USA")]
            row_parts = [p for p in row_parts if p is not None]
            a = {}
            for key in ["dom_int", "imp_int", "ttl_int", "ttl_fd",
                        "ttl_fd_c", "ttl_fd_g", "ttl_fd_i",
                        "dom_fd", "dom_expo", "dom_total", "imp_total"]:
                a[key] = sum(p[key] for p in row_parts)
            a["output"] = sum(p["output"] for p in row_parts)
            a["valu"] = sum(p["valu"] for p in row_parts)
            lab = labr[~labr.index.get_level_values(0).isin(["CHN", "USA"])]
            a["labr"] = lab.groupby("sec").sum()
            agg[reg] = a
        else:
            a = get(reg)
            a["labr"] = labr.loc[reg]
            agg[reg] = a

    # ---------- bilateral flows x0[origin, sector, destination] ----------
    # Control totals come from the IOT (dom_expo / imp_total) so that the
    # closure identity  C0 = VA_eff + (M - X)  holds exactly. Comtrade is
    # used ONLY for origin shares of each flow, not for levels.
    x0 = np.zeros((3, 3, 3))
    report = []

    # all-goods shares (proxy for services)
    g_chn_to_usa = t("CHN", "USA", "X", "prim") + t("CHN", "USA", "X", "manu")
    g_chn_wld = t("CHN", "WLD", "X", "prim") + t("CHN", "WLD", "X", "manu")
    g_usa_to_chn = t("USA", "CHN", "X", "prim") + t("USA", "CHN", "X", "manu")
    g_usa_wld = t("USA", "WLD", "X", "prim") + t("USA", "WLD", "X", "manu")

    for k, sec in enumerate(SECTORS):
        if sec == "serv":
            sh_chn_to_usa = g_chn_to_usa / g_chn_wld
            sh_usa_to_chn = g_usa_to_chn / g_usa_wld
        else:
            sh_chn_to_usa = t("CHN", "USA", "X", sec) / t("CHN", "WLD", "X", sec)
            sh_usa_to_chn = t("USA", "CHN", "X", sec) / t("USA", "WLD", "X", sec)

        expo_chn = float(agg["CHN"]["dom_expo"][sec])   # IOT control totals
        expo_usa = float(agg["USA"]["dom_expo"][sec])
        imp_chn = float(agg["CHN"]["imp_total"][sec])
        imp_usa = float(agg["USA"]["imp_total"][sec])

        x_chn_usa = expo_chn * sh_chn_to_usa
        x_usa_chn = expo_usa * sh_usa_to_chn

        x0[0, k, 1] = x_chn_usa
        x0[0, k, 2] = expo_chn - x_chn_usa              # CHN -> ROW
        x0[1, k, 0] = x_usa_chn
        x0[1, k, 2] = expo_usa - x_usa_chn              # USA -> ROW
        x0[2, k, 0] = imp_chn - x_usa_chn               # ROW -> CHN
        x0[2, k, 1] = imp_usa - x_chn_usa               # ROW -> USA

    # domestic + ROW-ROW residuals so that rows sum to gross output
    for oi, reg in enumerate(REGIONS):
        for k, sec in enumerate(SECTORS):
            y = float(agg[reg]["output"][sec])
            dom = y - x0[oi, k, :].sum()
            x0[oi, k, oi] = dom
            report.append((reg, sec, y, dom, dom / y))
    rep = pd.DataFrame(report, columns=["region", "sector", "gross_output",
                                        "domestic_flow", "domestic_share"])
    print("=== domestic flow residuals (must be > 0) ===")
    print(rep.round(3).to_string(index=False))
    assert (rep["domestic_flow"] > 0).all(), "negative domestic flow!"

    # ---------- raw io coefficients and final demand ----------
    io_rows, c0_rows = [], []
    for reg in REGIONS:
        a = agg[reg]
        for i_out in SECTORS:
            y = float(a["output"][i_out])
            for j_in in SECTORS:
                io_rows.append((j_in, reg, i_out,
                                float(a["ttl_int"].loc[j_in, i_out]) / y))
        for sec in SECTORS:
            c0_rows.append((reg, sec, float(a["ttl_fd_c"][sec]),
                            float(a["ttl_fd_g"][sec]),
                            float(a["ttl_fd_i"][sec])))
    io_df = pd.DataFrame(io_rows, columns=["input", "region", "output",
                                           "value"])
    c0_df = pd.DataFrame(c0_rows, columns=["region", "sector",
                                           "C0", "G0", "I0"])
    assert (c0_df["C0"] > 0).all()

    # ---------- exact reconciliation ----------
    # The raw data violate the use identity  sum_r x0*(1+tau0) = C0 + Int
    # by ~1-2% (CIF/FOB wedge between IOT import rows and partner EXPO
    # columns, plus the services origin-share proxy). Absorb it by scaling
    # each region's intermediate-input ROWS (per input sector) so that
    # intermediate USE of composite i in region s equals
    # (duty-inclusive inflow of i into s) - C0[s,i]. Capital (already a
    # residual containing NOPS+CFC+TXS, i.e. where import duties sit in
    # the IOT) absorbs the implied column-balance change, so benchmark
    # tariff revenue T0 = sum(tau0*x0) is consistently pulled out of the
    # capital residual rather than double counted.
    tau0_arr = np.zeros((3, 3, 3))
    tau_csv = OUT / "baseline_tariffs_2022.csv"
    if tau_csv.exists():
        tau_df = pd.read_csv(tau_csv)
        for row_ in tau_df.itertuples():
            tau0_arr[REGIONS.index(row_.origin), SECTORS.index(row_.sector),
                     REGIONS.index(row_.destination)] = row_.rate
        print("\n=== loaded baseline tariffs (nonzero entries) ===")
        print(tau_df[tau_df["rate"] > 0].round(4).to_string(index=False))
    T0 = {reg: sum(tau0_arr[oi, k, si] * x0[oi, k, si]
                   for oi in range(3) for k in range(3))
          for si, reg in enumerate(REGIONS)}

    c0_mat = c0_df.set_index(["region", "sector"])[["C0", "G0", "I0"]]
    for si, reg in enumerate(REGIONS):
        for k, sec in enumerate(SECTORS):
            inflow = sum(x0[oi, k, si] * (1 + tau0_arr[oi, k, si])
                         for oi in range(3))
            final_use = c0_mat.loc[reg, sec].sum()   # C0 + G0 + I0
            int_use = inflow - final_use
            raw_int = sum(float(agg[reg]["ttl_int"].loc[sec, j_out])
                          for j_out in SECTORS)
            phi = int_use / raw_int
            mask = (io_df["region"] == reg) & (io_df["input"] == sec)
            io_df.loc[mask, "value"] = io_df.loc[mask, "value"] * phi
            if abs(phi - 1) > 0.05:
                print(f"  WARNING: large io adjustment {reg}/{sec}: "
                      f"phi={phi:.4f}")

    # rebuild factor payments with exact column balance under adjusted io
    prod_rows = []
    for reg in REGIONS:
        a = agg[reg]
        for sec in SECTORS:
            y = float(a["output"][sec])
            lab = float(a["labr"][sec])
            inter = sum(
                float(io_df[(io_df["region"] == reg)
                            & (io_df["input"] == j_in)
                            & (io_df["output"] == sec)]["value"].iloc[0]) * y
                for j_in in SECTORS)
            cap = y - inter - lab
            prod_rows.append((reg, sec, y, lab, cap))
    prod_df = pd.DataFrame(prod_rows, columns=["region", "sector", "Y0",
                                               "labor", "capital"])
    assert (prod_df["capital"] > 0).all(), "negative capital residual!"

    flows = pd.DataFrame(
        [(REGIONS[o], SECTORS[k], REGIONS[s], x0[o, k, s])
         for o in range(3) for k in range(3) for s in range(3)],
        columns=["origin", "sector", "destination", "value"])
    print("\n=== bilateral flows x0 ===")
    print(flows.round(0).to_string(index=False))
    assert (flows["value"] > 0).all()

    # ---------- transfers and closure check ----------
    B = {}
    for si, reg in enumerate(REGIONS):
        imports = sum(x0[oi, k, si] for oi in range(3) if oi != si
                      for k in range(3))
        exports = sum(x0[si, k, dj] for dj in range(3) if dj != si
                      for k in range(3))
        B[reg] = imports - exports
    print("\n=== B = M - X ===")
    for reg in REGIONS:
        print(f"  {reg}: {B[reg]:,.0f}")
    print(f"  sum: {sum(B.values()):.6f}")

    # identity check: sum(C0+G0+I0)  vs  (labor + capital) + T0 + B
    # With the io-row reconciliation above this holds EXACTLY by construction
    # (duty-inclusive use per region = C0+G0+I0 + Int', and VA_eff + T0 + B).
    print("\n=== closure check: sum(C0+G0+I0) vs (labor+capital) + T0 + B ===")
    for reg in REGIONS:
        fd_tot = c0_df[c0_df["region"] == reg][["C0", "G0", "I0"]] \
            .sum().sum()
        va_eff = prod_df[prod_df["region"] == reg][["labor", "capital"]] \
            .sum().sum()
        print(f"  {reg}: FD={fd_tot:,.0f}  VA_eff+T0+B="
              f"{va_eff + T0[reg] + B[reg]:,.0f}  "
              f"(T0={T0[reg]:,.0f})  diff={fd_tot - va_eff - T0[reg] - B[reg]:,.2f}")

    io_df.to_csv(OUT / "io_coefficients_2022.csv", index=False)
    prod_df.to_csv(OUT / "production_2022.csv", index=False)
    c0_df.to_csv(OUT / "final_demand_2022.csv", index=False)
    flows.to_csv(OUT / "bilateral_flows_2022.csv", index=False)
    pd.DataFrame({"region": REGIONS,
                  "net_foreign_transfer": [B[r] for r in REGIONS]}
                 ).to_csv(OUT / "net_transfers_2022.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
