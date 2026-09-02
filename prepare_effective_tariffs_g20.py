# -*- coding: utf-8 -*-
"""
Effective-tariff correction factors for the 14-region model.

CHN/USA/JPN/KOR: GFS SOO main files (S13).
EU27: EC figure (see prepare_effective_tariffs.py).
New 8 (VNM THA MYS IDN SGP PHL IND MEX): GFS G1151 batch files
  (12_IMF_GFS/明细_G1151_ROW/, downloaded for the ROW correction),
  S13 -> S1311 -> S1311B fallback via live API when needed.
ROW: reused from data_real_6x12/tariff_effective_factors.csv.

Output: data_real_g20/tariff_effective_factors.csv
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(r"D:\数据")
OUT = ROOT / "simple_multicountry_cge" / "data_real_g20"
OUT.mkdir(parents=True, exist_ok=True)
GFS_MAIN = ROOT / "CGE公开数据" / "12_IMF_GFS" / "明细_按国家"
GFS_ROW = ROOT / "CGE公开数据" / "12_IMF_GFS" / "明细_G1151_ROW"
FX6 = ROOT / "simple_multicountry_cge" / "data_real_6x12" \
    / "tariff_effective_factors.csv"

FX = {"CHN": 6.73715811237119, "JPN": 131.498140443764,
      "KOR": 1291.44666666667, "USA": 1.0}
NEW8 = ["IND", "CAN", "MEX", "BRA", "ZAF", "RUS", "SAU", "AUS"]
EU27_G1151_MUSD = 25_000 / 0.75 * 1.0530
# CAN: not in IMF GFS (federal customs not in S1311). Canada Annual
# Financial Report FY2022-23: customs import duties = CAD 6,057 mn;
# CAD/USD 2022 = 1.3016 (World Bank PA.NUS.FCRF).
CAN_G1151_MUSD = 6_057 / 1.30155477474355

_FX_CACHE: dict = {}


def fetch_fx_all() -> dict:
    """One batched WDI call for all 8 expansion economies."""
    if _FX_CACHE:
        return _FX_CACHE
    codes = ";".join(NEW8)
    url = ("https://api.worldbank.org/v2/country/" + codes
           + "/indicator/PA.NUS.FCRF?date=2022&format=json&per_page=100")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
            d = r.json()
            out = {x["countryiso3code"]: float(x["value"])
                   for x in d[1] if x["value"] is not None}
            _FX_CACHE.update(out)
            return _FX_CACHE
        except Exception as e:
            print(f"  FX fetch attempt {attempt + 1} failed: {e}")
    return _FX_CACHE


def fetch_fx(iso3: str) -> float | None:
    return fetch_fx_all().get(iso3)


def g1151(cc: str, main: bool) -> float | None:
    """Local-currency G1151 for 2022; main files for the big 4,
    batch files + live fallback for the new 8."""
    f = (GFS_MAIN / f"GFS_SOO_{cc}_XDC.csv") if main else \
        (GFS_ROW / f"GFS_SOO_{cc}_G1151_XDC.csv")
    if f.exists():
        df = pd.read_csv(f)
        if "SECTOR" in df.columns:
            df = df[df["SECTOR"] == "S13"]
        if "INDICATOR" in df.columns:
            df = df[df["INDICATOR"] == "G1151_T"]
        v = df[df["TIME_PERIOD"] == 2022]["OBS_VALUE"].dropna()
        if len(v):
            return float(v.iloc[0])
    if main:
        return None
    for sector in ("S13", "S1311", "S1311B"):
        url = (f"https://api.imf.org/external/sdmx/2.1/data/GFS_SOO/"
               f"{cc}.{sector}..G1151_T.XDC.A?startPeriod=2022"
               f"&endPeriod=2022")
        try:
            r = requests.get(url, headers={"Accept": "text/csv"},
                             timeout=60)
            if r.status_code == 200 and "OBS_VALUE" in r.text:
                vals = pd.read_csv(io.StringIO(r.text))["OBS_VALUE"].dropna()
                if len(vals):
                    return float(vals.iloc[0])
        except Exception:
            pass
    return None


def main() -> None:
    tau = pd.read_csv(OUT / "baseline_tariffs_2022.csv")
    flows = pd.read_csv(OUT / "bilateral_flows_2022.csv")
    m = flows.merge(tau, on=["origin", "sector", "destination"], how="left")
    m["rate"] = m["rate"].fillna(0.0)
    t0 = (m["rate"] * m["value"]).groupby(m["destination"]).sum()

    old = pd.read_csv(FX6).set_index("region")
    rows = []
    for reg in ["CHN", "USA", "EU27", "JPN", "KOR", "IND", "CAN", "MEX",
                "BRA", "ZAF", "RUS", "SAU", "AUS", "ROW"]:
        model_t0 = float(t0.get(reg, 0.0))
        if reg in ("CHN", "USA", "JPN", "KOR"):
            actual = g1151(reg, main=True) / FX[reg] / 1e6
            factor = actual / model_t0 if model_t0 > 0 else 1.0
        elif reg == "EU27":
            actual = EU27_G1151_MUSD
            factor = actual / model_t0 if model_t0 > 0 else 1.0
        elif reg == "CAN":
            actual = CAN_G1151_MUSD
            factor = actual / model_t0 if model_t0 > 0 else 1.0
        elif reg == "ROW":
            factor = float(old.loc["ROW", "factor"])
            actual = float(old.loc["ROW", "g1151_mUSD"])
        else:
            rate = fetch_fx(reg)
            g = g1151(reg, main=False)
            if g is None or rate is None:
                print(f"{reg}: no G1151/FX -> factor 1.0")
                factor, actual = 1.0, None
            else:
                actual = g / rate / 1e6
                factor = actual / model_t0 if model_t0 > 0 else 1.0
        rows.append((reg, model_t0, actual, factor))
    out = pd.DataFrame(rows, columns=["region", "model_T0_mUSD",
                                      "g1151_mUSD", "factor"])
    out.to_csv(OUT / "tariff_effective_factors.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
