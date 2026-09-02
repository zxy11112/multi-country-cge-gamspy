# -*- coding: utf-8 -*-
"""
Compute the ROW effective-tariff correction factor:
  effective_rate = actual customs revenue (GFS G1151, 2022, USD)
                   / ROW total goods imports (IOT IMP rows, USD)
  factor         = effective_rate / model uniform rate (ROW mean AHS)

Covers the 49 ROW IOT economies; G1151 downloaded to
12_IMF_GFS/明细_G1151_ROW/ by download_g1151_row.py.
Exchange rates: World Bank PA.NUS.FCRF 2022 (fetched live).
Economies missing G1151 or FX are excluded from BOTH numerator and
denominator (imports scaled by coverage share).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(r"D:\数据")
OUT = ROOT / "simple_multicountry_cge" / "data_real_6x12"
GFS_DIR = ROOT / "CGE公开数据" / "12_IMF_GFS" / "明细_G1151_ROW"
DOMIMP_ZIP = ROOT / "CGE公开数据" / "06_OECD_IOT" / "IOTs_DOMIMP.zip"
BASE_TARIFFS = OUT / "baseline_tariffs_2022.csv"

ROW = ("AGO ARE ARG AUS BGD BLR BRA BRN CAN CHE CHL CIV CMR COD COL CRI "
       "EGY GBR HKG IDN IND ISL ISR JOR KAZ KHM LAO MAR MEX MMR MYS NGA "
       "NOR NZL PAK PER PHL RUS SAU SEN SGP STP THA TUN TUR TWN UKR VNM ZAF").split()


def fetch_fx() -> dict:
    """Official exchange rate 2022, LCU per USD, for all ROW economies."""
    codes = ";".join(ROW)
    url = ("https://api.worldbank.org/v2/country/" + codes
           + "/indicator/PA.NUS.FCRF?date=2022&format=json&per_page=500")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = json.loads(r.text)
    return {d["countryiso3code"]: d["value"] for d in data[1]
            if d["value"] is not None}


def g1151_local(cc: str) -> float | None:
    f = GFS_DIR / f"GFS_SOO_{cc}_G1151_XDC.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    v = df[df["TIME_PERIOD"] == 2022]["OBS_VALUE"]
    if len(v) == 0 or pd.isna(v.iloc[0]):
        return None
    return float(v.iloc[0])


def g1151_with_fallback(cc: str) -> float | None:
    """S13 local file first; then live API fallback S1311 / S1311B."""
    v = g1151_local(cc)
    if v is not None:
        return v
    for sector in ("S1311", "S1311B"):
        url = (f"https://api.imf.org/external/sdmx/2.1/data/GFS_SOO/"
               f"{cc}.{sector}..G1151_T.XDC.A?startPeriod=2022"
               f"&endPeriod=2022")
        try:
            r = requests.get(url, headers={"Accept": "text/csv"},
                             timeout=60)
            if r.status_code == 200 and "OBS_VALUE" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                vals = df["OBS_VALUE"].dropna()
                if len(vals):
                    return float(vals.iloc[0])
        except Exception:
            pass
    return None


def row_goods_imports() -> pd.Series:
    """IOT total imports (IMP rows) per ROW economy, million USD, 2022."""
    out = {}
    with zipfile.ZipFile(DOMIMP_ZIP) as zf:
        for cc in ROW:
            name = f"{cc}2022dom.csv"
            if name not in zf.namelist():
                continue
            with zf.open(name) as fh:
                df = pd.read_csv(fh, index_col=0)
            df.index = df.index.str.strip('"')
            imp_rows = [i for i in df.index
                        if i.startswith("IMP_") and i != "IMP_OTHER"]
            out[cc] = float(df.loc[imp_rows, "TOTAL"].sum())
    return pd.Series(out)


def main() -> None:
    fx = fetch_fx()
    print(f"FX fetched for {len(fx)} economies")
    imp = row_goods_imports()

    num = 0.0          # actual customs revenue, million USD
    den = 0.0          # imports of covered economies, million USD
    missing = []
    for cc in ROW:
        g = g1151_with_fallback(cc)
        rate = fx.get(cc)
        if g is None or rate is None or rate == 0:
            missing.append(cc)
            continue
        num += g / rate / 1e6
        den += imp.get(cc, 0.0)
    print(f"covered: {len(ROW) - len(missing)}/{len(ROW)}; "
          f"missing: {' '.join(missing)}")

    # model uniform ROW rate (mean AHS from baseline tariffs)
    tau = pd.read_csv(BASE_TARIFFS)
    model_rate = float(tau[(tau["destination"] == "ROW")
                           & (tau["origin"] != "ROW")
                           & (tau["sector"] != "SRV")]["rate"].mean())
    effective_rate = num / den
    factor = effective_rate / model_rate
    print(f"\nROW actual customs revenue (covered): {num:,.0f} mn USD")
    print(f"ROW goods imports (covered):        {den:,.0f} mn USD")
    print(f"effective rate: {effective_rate:.4f}  vs model {model_rate:.4f}")
    print(f"correction factor: {factor:.4f}")

    # update the factors file
    fpath = OUT / "tariff_effective_factors.csv"
    fac = pd.read_csv(fpath)
    fac.loc[fac["region"] == "ROW", ["g1151_mUSD", "factor"]] = [num, factor]
    fac.to_csv(fpath, index=False)
    print(f"\nupdated {fpath}")
    print(fac.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
