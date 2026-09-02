# -*- coding: utf-8 -*-
"""
Download WITS/TRAINS HS6 MFN tariffs (2022, partner=World) for the 8
expansion countries and parse to tidy CSV.

WITS uses M49 numeric codes (note: Comtrade's India is 699, WITS/M49 is 356):
  VNM 704, THA 764, MYS 458, IDN 360, SGP 702, PHL 608, IND 356, MEX 484

URL pattern (returns SDMX GenericData XML):
  .../DF_WITS_Tariff_TRAINS/.{code}.000..reported/?startperiod=2022&endperiod=2022

Raw XML saved to 10_WorldBank_WITS/HS6关税表_raw/ (convention),
tidy CSV to 整理/02_清洗/  as clean_tariff_wits_hs6_2022_expansion.csv
(same columns as clean_tariff_wits_hs6_2022_v1.csv).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(r"D:\数据")
RAW = ROOT / "CGE公开数据" / "10_WorldBank_WITS" / "HS6关税表_raw"
OUT = ROOT / "整理" / "02_清洗"
RAW.mkdir(parents=True, exist_ok=True)

COUNTRIES = {"VNM": ("704", "越南"), "THA": ("764", "泰国"),
             "MYS": ("458", "马来西亚"), "IDN": ("360", "印尼"),
             "SGP": ("702", "新加坡"), "PHL": ("608", "菲律宾"),
             "IND": ("356", "印度"), "MEX": ("484", "墨西哥")}

URL = ("https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
       "DF_WITS_Tariff_TRAINS/.{code}.000..reported/"
       "?startperiod=2022&endperiod=2022")

NS = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
      "m": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message"}


def parse(xml_path: Path, iso: str, name: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    rows = []
    for series in tree.iter():
        if not series.tag.endswith("}Series"):
            continue
        # SeriesKey: only direct Value children
        key = {}
        for sk in series:
            if sk.tag.endswith("}SeriesKey"):
                key = {v.get("id"): v.get("value") for v in sk
                       if v.tag.endswith("}Value")}
        prod = key.get("PRODUCTCODE")
        for obs in series:
            if not obs.tag.endswith("}Obs"):
                continue
            val = None
            attrs = {}
            for child in obs:
                if child.tag.endswith("}ObsValue"):
                    val = child.get("value")
                elif child.tag.endswith("}Attributes"):
                    attrs = {v.get("id"): v.get("value") for v in child
                             if v.tag.endswith("}Value")}
            tariff = attrs.get("TARIFFTYPE")
            nomen = attrs.get("NOMENCODE")
            rows.append({
                "source": "wits_hs6", "reporter_iso3": iso,
                "reporter_name": name, "year": 2022,
                "hs_code": prod, "hs_version": nomen or "",
                "tariff_type": tariff or "MFN", "duty_value": val,
                "duty_unit": "ad_valorem", "product_name": "",
                "partner_m49": "000", "partner_name": "World",
                "nomen_code": nomen or "", "tariff_type_raw": tariff or "",
                "sum_of_rates": attrs.get("SUM_OF_RATES", ""),
                "min_rate": attrs.get("MIN_RATE", ""),
                "max_rate": attrs.get("MAX_RATE", ""),
                "total_lines": attrs.get("TOTALNOOFLINES", ""),
                "pref_lines": attrs.get("NBR_PREF_LINES", ""),
                "mfn_lines": attrs.get("NBR_MFN_LINES", ""),
                "na_lines": attrs.get("NBR_NA_LINES", ""),
                "measure": "SimpleAverage", "note": "",
            })
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for iso, (code, name) in COUNTRIES.items():
        raw = RAW / f"WITS_HS6关税表_{name}_{int(code):03d}_2022.xml"
        if not raw.exists() or raw.stat().st_size < 100000:
            r = requests.get(URL.format(code=code), timeout=600)
            if r.status_code != 200:
                print(f"{iso}: http {r.status_code}")
                continue
            raw.write_bytes(r.content)
            print(f"{iso}: downloaded {len(r.content):,} bytes")
            time.sleep(5)
        df = parse(raw, iso, name)
        print(f"{iso}: parsed {len(df):,} HS6 lines, "
              f"nomen={df['nomen_code'].unique()}")
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        f = OUT / "clean_tariff_wits_hs6_2022_expansion.csv"
        out.to_csv(f, index=False)
        print(f"\nwritten {f} ({len(out):,} rows)")
    print("DONE")


if __name__ == "__main__":
    main()
