# -*- coding: utf-8 -*-
"""
Build a real-data benchmark WITH intermediate inputs for the
6-region (CHN / USA / JPN / KOR / EU27 / ROW) x 12-sector CGE model,
benchmark year 2022.

Sources (same pipeline as the 3-region version, generalized):
  - OECD harmonised national IOTs (DOMIMP + VA components), 2025 release
  - Cleaned UN Comtrade SITC long table (5 reporters: CHN USA JPN KOR EU27)
  - data_real_6x12/baseline_tariffs_2022.csv (from prepare_baseline_tariffs_6x12.py)

Construction conventions (identical to the 3-region version):
  - EU27 = sum of the 27 member states' IOT tables; EU27 reporter's
    Comtrade flows are extra-EU trade, intra-EU trade lands in the
    domestic residual.
  - Bilateral flows among the 5 explicit regions: exporter-side
    Comtrade records (FOB). Flows into an explicit region s from ROW:
    s's import records (CIF mirror) minus the other four explicit
    origins. Domestic and ROW->ROW flows are residuals that make each
    origin-sector's flows sum to gross output.
  - SRV has no Comtrade coverage: origin shares proxied by each
    pair's all-goods shares; levels from IOT DOM_EXPO / IMP rows.
  - io reconciliation: per (region, input sector) scale factors phi so
    that intermediate use equals duty-inclusive inflow minus final
    demand (C0+G0+I0). Capital is the column-balance residual.
  - Closure  C0+G0+I0 = VA_eff + T0 + B  holds exactly by construction.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
IOT_DIR = ROOT / "CGE公开数据" / "06_OECD_IOT"
DOMIMP_ZIP = IOT_DIR / "IOTs_DOMIMP.zip"
VA_ZIP = IOT_DIR / "IOTs_VA.zip"
COMTRADE = ROOT / "整理" / "02_清洗" / "clean_comtrade_sitc_1992-2023_v1.1.csv"
OUT = ROOT / "simple_multicountry_cge" / "data_real_6x12"

YEAR = 2022
REGIONS = ["CHN", "USA", "JPN", "KOR", "EU27", "ROW"]
SECTORS = ["AGF", "MIN", "ENR", "CHM", "TXL", "WDP",
           "NMM", "MAC", "ELE", "VEH", "OTM", "SRV"]
R5 = REGIONS[:-1]                      # explicit regions (Comtrade reporters)
EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
        "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
        "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]

ISIC_MAP = {
    "AGF": ["A01", "A02", "A03", "C10T12"],
    "MIN": ["B05", "B07", "B08", "B09"],
    "ENR": ["B06", "C19"],
    "CHM": ["C20", "C21", "C22"],
    "TXL": ["C13T15"],
    "WDP": ["C16", "C17_18"],
    "NMM": ["C23", "C24A", "C24B", "C25"],
    "MAC": ["C28"],
    "ELE": ["C26", "C27"],
    "VEH": ["C29", "C301", "C302T309"],
    "OTM": ["C31T33"],
    "SRV": ["D", "E", "F", "G", "H49", "H50", "H51", "H52", "H53", "I",
            "J58T60", "J61", "J62_63", "K", "L", "M", "N", "O", "P", "Q",
            "R", "S", "T"],
}

FD_C = ["HFCE", "NPISH", "DPABR"]
FD_G = ["GGFC"]
FD_I = ["GFCF", "INVNT"]
FD_COLS = FD_C + FD_G + FD_I

_cache: dict[str, dict | None] = {}


def sector_of(ind: str) -> str:
    for sec, inds in ISIC_MAP.items():
        if ind in inds:
            return sec
    raise KeyError(ind)


def sitc2_sector(code: str) -> str | None:
    """SITC 2-digit -> model sector; None = excluded from the trade system.

    Excluded: 9x (special transactions 93, coin 96, gold 97) -- large
    flows with no counterpart in production accounts.
    28 (ores & scrap) is assigned to NMM: scrap-metal exports of JPN/KOR
    exceed their mining output, so it cannot stay in MIN.
    87/88 (scientific/photo instruments) belong to C26 -> ELE.
    """
    if code[0] == "9":
        return None
    if code[0] in "014" or code in ("21", "22", "26", "29"):
        return "AGF"
    if code == "27":
        return "MIN"
    if code[0] == "3":
        return "ENR"
    if code[0] == "5" or code in ("23", "62"):
        return "CHM"
    if code in ("61", "65", "83", "84", "85"):
        return "TXL"
    if code in ("24", "25", "63", "64"):
        return "WDP"
    if code in ("28", "66", "67", "68", "69"):
        return "NMM"
    if code in ("71", "72", "73", "74"):
        return "MAC"
    if code in ("75", "76", "77", "87", "88"):
        return "ELE"
    if code in ("78", "79"):
        return "VEH"
    return "OTM"


def load_dom(country: str) -> dict | None:
    name = f"{country}{YEAR}dom.csv"
    with zipfile.ZipFile(DOMIMP_ZIP) as zf:
        if name not in zf.namelist():
            return None
        with zf.open(name) as fh:
            df = pd.read_csv(fh, index_col=0)
    df.index = df.index.str.strip('"')
    df.columns = df.columns.str.strip('"')

    ind_cols = [c for c in df.columns
                if c not in FD_COLS + ["CONS_NONRES", "EXPO", "IMPO",
                                       "TOTAL", ".."]]
    sec_cols = {s: [c for c in ind_cols if sector_of(c) == s]
                for s in SECTORS}

    def block(prefix: str) -> pd.DataFrame:
        rows = df.loc[[i for i in df.index
                       if i.startswith(prefix) and "OTHER" not in i]].copy()
        rows["sec"] = [sector_of(i[len(prefix):]) for i in rows.index]
        return rows.groupby("sec").sum().reindex(SECTORS).fillna(0.0)

    out = {}
    for pfx, key in [("DOM_", "dom"), ("IMP_", "imp")]:
        blk = block(pfx)
        out[key + "_int"] = pd.DataFrame(
            {s: blk[sec_cols[s]].sum(axis=1) for s in SECTORS})
        for tag, cols in [("c", FD_C), ("g", FD_G), ("i", FD_I)]:
            out[key + "_fd_" + tag] = blk[cols].sum(axis=1)
        out[key + "_fd"] = blk[FD_COLS].sum(axis=1)
        out[key + "_expo"] = blk["EXPO"] + blk["CONS_NONRES"]
        out[key + "_total"] = blk["TOTAL"]
    out["ttl_int"] = out["dom_int"] + out["imp_int"]
    for tag in ["c", "g", "i", ""]:
        suffix = ("_" + tag) if tag else ""
        out["ttl_fd" + suffix] = (out["dom_fd" + suffix]
                                  + out["imp_fd" + suffix])
    out["output"] = df.loc["OUTPUT", ind_cols].groupby(
        [sector_of(c) for c in ind_cols]).sum().reindex(SECTORS).fillna(0.0)
    out["valu"] = df.loc["VALU", ind_cols].groupby(
        [sector_of(c) for c in ind_cols]).sum().reindex(SECTORS).fillna(0.0)
    return out


def get(country: str) -> dict | None:
    if country not in _cache:
        _cache[country] = load_dom(country)
    return _cache[country]


def aggregate(countries: list[str], labr: pd.Series) -> dict:
    parts = [get(c) for c in countries]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise RuntimeError(f"no IOT data for {countries}")
    a = {}
    keys = [k for k in parts[0] if k != "labr"]
    for key in keys:
        a[key] = sum(p[key] for p in parts)
    lab = labr[labr.index.get_level_values(0).isin(countries)]
    a["labr"] = lab.groupby("sec").sum().reindex(SECTORS).fillna(0.0)
    return a


def load_labr() -> pd.Series:
    with zipfile.ZipFile(VA_ZIP) as zf:
        with zf.open("VAcomponents.csv") as fh:
            va = pd.read_csv(fh)
    va = va[(va["year"] == YEAR) & (va["variable"] == "LABR")]
    va["sec"] = va["industry"].map(sector_of)
    return va.groupby(["country", "sec"])["mlln_USD"].sum()


def load_trade() -> pd.Series:
    cols = ["reporter_iso3", "partner_iso3", "flow_code", "cmd_code",
            "period", "cl_code", "primary_value"]
    use = pd.read_csv(
        COMTRADE, usecols=cols,
        dtype={"reporter_iso3": str, "partner_iso3": str,
               "flow_code": str, "cmd_code": str})
    use = use[(use["period"] == YEAR) & (use["cl_code"] == "S4")
              & (use["cmd_code"] != "TOTAL")
              & (use["reporter_iso3"].isin(R5))]
    use = use[use["cmd_code"].str.len() == 2]
    use["sec"] = use["cmd_code"].map(sitc2_sector)
    n_excl = use["sec"].isna().sum()
    if n_excl:
        excl_val = use.loc[use["sec"].isna(), "primary_value"].sum() / 1e9
        print(f"excluded SITC 9x (special/gold): {n_excl} records, "
              f"{excl_val:.0f} bn USD")
    use = use.dropna(subset=["sec"])
    use = use[use["partner_iso3"].isin(R5 + ["WLD"])
              & use["flow_code"].isin(["X", "M"])]
    g = (use.groupby(["reporter_iso3", "partner_iso3", "flow_code", "sec"])
            ["primary_value"].sum() / 1e6)
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labr = load_labr()
    tr = load_trade()
    nR, nS = len(REGIONS), len(SECTORS)

    def t(rep, par, flow, sec):
        try:
            return float(tr[(rep, par, flow, sec)])
        except KeyError:
            return 0.0

    with zipfile.ZipFile(DOMIMP_ZIP) as zf:
        all_countries = sorted({n[:3] for n in zf.namelist()
                                if n.endswith(f"{YEAR}dom.csv")})
    row_countries = [c for c in all_countries
                     if c not in EU27 + ["CHN", "USA", "JPN", "KOR"]]
    print(f"IOT economies: {len(all_countries)}; EU27 members found: "
          f"{len([c for c in EU27 if c in all_countries])}; "
          f"ROW: {len(row_countries)}")

    agg: dict[str, dict] = {}
    for reg in ["CHN", "USA", "JPN", "KOR"]:
        agg[reg] = aggregate([reg], labr)
    agg["EU27"] = aggregate(EU27, labr)
    agg["ROW"] = aggregate(row_countries, labr)

    # ---------- bilateral flows x0[origin, sector, destination] ----------
    # Control totals: exporter side = IOT dom_expo, importer side = IOT
    # imp_total; Comtrade provides only the origin/destination SHARES.
    # Exception EU27: IOT totals include intra-EU trade, so for goods we
    # use Comtrade WLD totals (= extra-EU trade) as control totals, and
    # scale services by the same extra-EU ratio.
    x0 = np.zeros((nR, nS, nR))

    goods_sectors = SECTORS[:-1]
    # BaTIS bilateral services flows (balanced values), if available:
    # used for SRV destination shares and for EU27 extra-EU SRV totals.
    srv_csv = OUT / "services_flows_batis_2022.csv"
    srv_x = None
    if srv_csv.exists():
        srv_df = pd.read_csv(srv_csv)
        srv_x = np.zeros((nR, nR))
        for row_ in srv_df.itertuples():
            srv_x[REGIONS.index(row_.origin),
                  REGIONS.index(row_.destination)] = row_.value
        print("loaded BaTIS services flows "
              f"(total {srv_x.sum() / 1e3:,.0f} bn USD)")

    for o in R5:
        iot_goods_expo = sum(float(agg[o]["dom_expo"][sec])
                             for sec in goods_sectors)
        ct_goods_expo = sum(t(o, "WLD", "X", sec) for sec in goods_sectors)
        iot_goods_imp = sum(float(agg[o]["imp_total"][sec])
                            for sec in goods_sectors)
        ct_goods_imp = sum(t(o, "WLD", "M", sec) for sec in goods_sectors)
        if o == "EU27":
            ratio_x = (ct_goods_expo / iot_goods_expo
                       if iot_goods_expo > 0 else 1.0)
            ratio_m = (ct_goods_imp / iot_goods_imp
                       if iot_goods_imp > 0 else 1.0)
            print(f"EU27 extra-EU ratios: exports {ratio_x:.3f}, "
                  f"imports {ratio_m:.3f}")
        for k, sec in enumerate(SECTORS):
            goods = sec != "SRV"
            if o == "EU27" and not goods and srv_x is not None:
                # actual extra-EU services trade from BaTIS
                expo_o = float(srv_x[REGIONS.index("EU27"), :].sum())
                imp_o = float(srv_x[:, REGIONS.index("EU27")].sum())
            elif o == "EU27":
                expo_o = (t(o, "WLD", "X", sec) if goods else
                          float(agg[o]["dom_expo"][sec]) * ratio_x)
                imp_o = (t(o, "WLD", "M", sec) if goods else
                         float(agg[o]["imp_total"][sec]) * ratio_m)
            else:
                expo_o = float(agg[o]["dom_expo"][sec])
                imp_o = float(agg[o]["imp_total"][sec])
            agg[o].setdefault("_expo", {})[sec] = expo_o
            agg[o].setdefault("_imp", {})[sec] = imp_o

    for k, sec in enumerate(SECTORS):
        goods = sec != "SRV"
        for oi, o in enumerate(R5):
            expo_o = agg[o]["_expo"][sec]
            # destination shares from Comtrade export records
            if goods:
                denom = t(o, "WLD", "X", sec)
                sh = {s_: (t(o, s_, "X", sec) / denom if denom > 0 else 0.0)
                      for s_ in R5 if s_ != o}
            elif srv_x is not None:
                # services: BaTIS bilateral shares (actual services data)
                oi_b = REGIONS.index(o)
                denom = srv_x[oi_b, :].sum()
                sh = {s_: (srv_x[oi_b, REGIONS.index(s_)] / denom
                           if denom > 0 else 0.0)
                      for s_ in R5 if s_ != o}
            else:
                # services fallback: proxy by all-goods shares
                denom = sum(t(o, "WLD", "X", gg) for gg in goods_sectors)
                sh = {s_: (sum(t(o, s_, "X", gg) for gg in goods_sectors)
                           / denom if denom > 0 else 0.0)
                      for s_ in R5 if s_ != o}
            sh_row = max(1.0 - sum(sh.values()), 0.0)
            for s_, frac in sh.items():
                x0[oi, k, REGIONS.index(s_)] = expo_o * frac
            x0[oi, k, -1] = expo_o * sh_row                # -> ROW
        # ROW -> explicit region s
        for si, s_ in enumerate(R5):
            imp_s = agg[s_]["_imp"][sec]
            from_explicit = sum(x0[oi, k, si] for oi in range(len(R5)))
            x0[-1, k, si] = imp_s - from_explicit

    # floor: CES calibration needs strictly positive flows
    n_nonpos = int((x0 < 1.0).sum())
    x0 = np.maximum(x0, 1.0)
    print(f"bilateral flows: {nR*nS*nR} cells, "
          f"{n_nonpos} floored to 1.0 (million USD)")

    # domestic + ROW->ROW residuals so origin rows sum to gross output
    report = []
    for oi, reg in enumerate(REGIONS):
        for k, sec in enumerate(SECTORS):
            y = float(agg[reg]["output"][sec])
            dom = y - x0[oi, k, :].sum()
            x0[oi, k, oi] = dom
            report.append((reg, sec, y, dom))
    rep = pd.DataFrame(report, columns=["region", "sector", "gross_output",
                                        "domestic_flow"])
    bad = rep[rep["domestic_flow"] <= 0]
    if len(bad):
        print("WARNING: non-positive domestic residuals:")
        print(bad.to_string(index=False))
        raise SystemExit(1)
    n_small = (rep["domestic_flow"] / rep["gross_output"] < 0.02).sum()
    print(f"domestic residuals: {len(rep)} ok, "
          f"{n_small} below 2% of output")

    # ---------- final demand, io coefficients, baseline tariffs ----------
    c0_rows, io_rows = [], []
    for reg in REGIONS:
        a = agg[reg]
        for sec in SECTORS:
            c0_rows.append((reg, sec, float(a["ttl_fd_c"][sec]),
                            float(a["ttl_fd_g"][sec]),
                            float(a["ttl_fd_i"][sec])))
        for i_out in SECTORS:
            y = float(a["output"][i_out])
            for j_in in SECTORS:
                io_rows.append((j_in, reg, i_out,
                                float(a["ttl_int"].loc[j_in, i_out]) / y))
    c0_df = pd.DataFrame(c0_rows, columns=["region", "sector",
                                           "C0", "G0", "I0"])
    io_df = pd.DataFrame(io_rows, columns=["input", "region", "output",
                                           "value"])
    for col in ["C0", "G0", "I0"]:
        n_neg = (c0_df[col] <= 0).sum()
        if n_neg:
            print(f"WARNING: {n_neg} non-positive {col} cells; "
                  f"clipped to small positive")
            c0_df[col] = c0_df[col].clip(lower=1.0)

    tau0_arr = np.zeros((nR, nS, nR))
    tau_csv = OUT / "baseline_tariffs_2022.csv"
    if tau_csv.exists():
        for row_ in pd.read_csv(tau_csv).itertuples():
            tau0_arr[REGIONS.index(row_.origin), SECTORS.index(row_.sector),
                     REGIONS.index(row_.destination)] = row_.rate
    # effective-tariff correction: anchor T0 to actual customs revenue
    # (IMF GFS G1151); see prepare_effective_tariffs.py
    fac_csv = OUT / "tariff_effective_factors.csv"
    if fac_csv.exists():
        fac = pd.read_csv(fac_csv).set_index("region")["factor"]
        for si, reg in enumerate(REGIONS):
            tau0_arr[:, :, si] *= float(fac.get(reg, 1.0))
        print("\n=== effective-tariff factors applied ===")
        print(fac.round(3).to_string())
        # write the effective rates for the model loader
        rows = [(REGIONS[o], SECTORS[k], REGIONS[s], tau0_arr[o, k, s])
                for o in range(nR) for k in range(nS) for s in range(nR)]
        pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                    "rate"]).to_csv(
            OUT / "baseline_tariffs_effective_2022.csv", index=False)
    T0 = {reg: sum(tau0_arr[oi, k, si] * x0[oi, k, si]
                   for oi in range(nR) for k in range(nS))
          for si, reg in enumerate(REGIONS)}

    # ---------- io reconciliation ----------
    c0_mat = c0_df.set_index(["region", "sector"])[["C0", "G0", "I0"]]
    phis = []
    for si, reg in enumerate(REGIONS):
        for k, sec in enumerate(SECTORS):
            inflow = sum(x0[oi, k, si] * (1 + tau0_arr[oi, k, si])
                         for oi in range(nR))
            int_use = inflow - c0_mat.loc[reg, sec].sum()
            raw_int = sum(float(agg[reg]["ttl_int"].loc[sec, j_out])
                          for j_out in SECTORS)
            phi = int_use / raw_int
            phis.append(phi)
            mask = (io_df["region"] == reg) & (io_df["input"] == sec)
            io_df.loc[mask, "value"] = io_df.loc[mask, "value"] * phi
    phis = np.array(phis)
    print(f"io adjustment phi: min {phis.min():.4f}  max {phis.max():.4f}  "
          f"mean {phis.mean():.4f}")

    # ---------- factor payments with exact column balance ----------
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
            prod_rows.append((reg, sec, y, lab, y - inter - lab))
    prod_df = pd.DataFrame(prod_rows, columns=["region", "sector", "Y0",
                                               "labor", "capital"])
    n_negcap = (prod_df["capital"] <= 0).sum()
    if n_negcap:
        print("WARNING: non-positive capital residuals:")
        print(prod_df[prod_df["capital"] <= 0].to_string(index=False))
    # floor tiny/negative capital residuals; shift the shortfall to labor
    # so the column balance (labor + capital) is preserved exactly
    small = prod_df["capital"] < 1.0
    if small.any():
        shortfall = 1.0 - prod_df.loc[small, "capital"]
        prod_df.loc[small, "labor"] -= shortfall
        prod_df.loc[small, "capital"] = 1.0
        print(f"capital floored in {int(small.sum())} cell(s) "
              f"(shortfall shifted to labor)")

    # ---------- transfers & closure ----------
    flows = pd.DataFrame(
        [(REGIONS[o], SECTORS[k], REGIONS[s], x0[o, k, s])
         for o in range(nR) for k in range(nS) for s in range(nR)],
        columns=["origin", "sector", "destination", "value"])
    n_nonpos = (flows["value"] <= 0).sum()
    print(f"bilateral flows: {len(flows)} cells, {n_nonpos} non-positive")

    B = {}
    for si, reg in enumerate(REGIONS):
        imports = sum(x0[oi, k, si] for oi in range(nR) if oi != si
                      for k in range(nS))
        exports = sum(x0[si, k, dj] for dj in range(nR) if dj != si
                      for k in range(nS))
        B[reg] = imports - exports

    print("\n=== closure check: sum(C0+G0+I0) vs (labor+capital)+T0+B ===")
    for reg in REGIONS:
        fd_tot = c0_df[c0_df["region"] == reg][["C0", "G0", "I0"]].sum().sum()
        va_eff = prod_df[prod_df["region"] == reg][["labor", "capital"]] \
            .sum().sum()
        print(f"  {reg}: FD={fd_tot:,.0f}  VA+T0+B="
              f"{va_eff + T0[reg] + B[reg]:,.0f}  "
              f"diff={fd_tot - va_eff - T0[reg] - B[reg]:,.2f}")
    print("\n=== B = M - X (million USD) ===")
    for reg in REGIONS:
        print(f"  {reg}: {B[reg]:,.0f}")
    print(f"  sum: {sum(B.values()):.6f}")

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
