# -*- coding: utf-8 -*-
"""
Download WITS/TRAINS HS6 MFN tariffs (2006-2019, partner=World) for the
G20 reporter set and parse to per-year tidy CSVs
(same schema as clean_tariff_wits_hs6_2005_g20.csv).

Whole-country XMLs sometimes get truncated by server timeouts; per-chapter
requests (HS 2-digit) are small and reliable, so truncated countries are
re-tried chapter by chapter and merged.

Resume-safe: existing raw XMLs that parse cleanly are skipped, so the script
can be re-run after an interruption.

Usage: python download_wits_yearly.py [--years 2006 2007 ...]
Output: 整理/02_清洗/clean_tariff_wits_hs6_<year>_g20.csv
"""
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(r"D:\数据\数据")
RAW = ROOT / "CGE公开数据" / "10_WorldBank_WITS" / "HS6关税表_raw"
OUT = ROOT / "整理" / "02_清洗"
RAW.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2006, 2020))
COUNTRIES = {"AUS": ("036", "澳大利亚"), "BRA": ("076", "巴西"),
             "CAN": ("124", "加拿大"), "CHN": ("156", "中国"),
             "DEU": ("276", "德国"), "IND": ("356", "印度"),
             "JPN": ("392", "日本"), "KOR": ("410", "韩国"),
             "MEX": ("484", "墨西哥"), "RUS": ("643", "俄罗斯"),
             "SAU": ("682", "沙特"), "USA": ("840", "美国"),
             "ZAF": ("710", "南非")}

URL = ("https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
       "DF_WITS_Tariff_TRAINS/.{code}.000..reported/"
       "?startperiod={year}&endperiod={year}")
CH_URL = ("https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
          "DF_WITS_Tariff_TRAINS/.{code}.000.{ch}.reported/"
          "?startperiod={year}&endperiod={year}")
CHAPTERS = [f"{i:02d}" for i in range(1, 98)]

HDR = ('<?xml version="1.0" encoding="utf-8"?>'
       '<message:GenericData xmlns:footer="http://www.sdmx.org/resources'
       '/sdmxml/schemas/v2_1/message/footer" xmlns:generic="http://www.sdmx.'
       'org/resources/sdmxml/schemas/v2_1/data/generic" xmlns:message="http:'
       '//www.sdmx.org/resources/sdmxml/schemas/v2_1/message" xmlns:common='
       '"http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common" xmlns:'
       'xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xml="http://'
       'www.w3.org/XML/1999/XML"><message:Header></message:Header>'
       '<message:DataSet>')
FTR = "</message:DataSet></message:GenericData>"


def parses_ok(path: Path) -> bool:
    try:
        ET.parse(path)
        return True
    except (ET.ParseError, OSError):
        return False


def extract_series(xml_text: str) -> list[str]:
    out, i = [], 0
    while True:
        a = xml_text.find("<generic:Series>", i)
        if a < 0:
            break
        b = xml_text.find("</generic:Series>", a)
        if b < 0:
            break
        out.append(xml_text[a:b + len("</generic:Series>")])
        i = b + 1
    return out


def ensure_raw(code: str, name: str, year: int) -> Path | None:
    raw = RAW / f"WITS_HS6关税表_{name}_{int(code):03d}_{year}.xml"
    if raw.exists() and parses_ok(raw):
        return raw
    try:
        r = requests.get(URL.format(code=code, year=year), timeout=900)
        if r.status_code != 200:
            print(f"    {name} {year}: http {r.status_code}; chapter-wise",
                  flush=True)
        else:
            raw.write_bytes(r.content)
    except Exception as e:  # noqa: BLE001
        print(f"    {name} {year}: whole-country failed ({e}); "
              f"chapter-wise", flush=True)
    if raw.exists() and parses_ok(raw):
        return raw
    # chapter-wise fallback
    series_all = []
    for ch in CHAPTERS:
        for attempt in range(3):
            try:
                txt = requests.get(CH_URL.format(code=code, ch=ch, year=year),
                                   timeout=300).text
                series_all.extend(extract_series(txt))
                break
            except Exception as e:  # noqa: BLE001
                print(f"    {name} {year} ch{ch}: {e} "
                      f"(attempt {attempt + 1})", flush=True)
                time.sleep(5)
        time.sleep(0.5)
    raw.write_text(HDR + "".join(series_all) + FTR, encoding="utf-8")
    if parses_ok(raw):
        print(f"    {name} {year}: rebuilt chapter-wise "
              f"({len(series_all)} series)", flush=True)
        return raw
    print(f"    {name} {year}: FAILED even chapter-wise", flush=True)
    return None


def parse(xml_path: Path, iso: str, name: str, year: int) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    rows = []
    for series in tree.iter():
        if not series.tag.endswith("}Series"):
            continue
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
            rows.append({
                "source": "wits_hs6", "reporter_iso3": iso,
                "reporter_name": name, "year": year,
                "hs_code": prod, "hs_version": attrs.get("NOMENCODE", ""),
                "tariff_type": attrs.get("TARIFFTYPE", "MFN"),
                "duty_value": val, "duty_unit": "ad_valorem",
                "product_name": "", "partner_m49": "000",
                "partner_name": "World",
                "nomen_code": attrs.get("NOMENCODE", ""),
                "tariff_type_raw": attrs.get("TARIFFTYPE", ""),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=YEARS)
    args = ap.parse_args()
    for year in args.years:
        frames = []
        for iso, (code, name) in COUNTRIES.items():
            raw = ensure_raw(code, name, year)
            if raw is None:
                print(f"  {iso} {year}: SKIPPED (no data)", flush=True)
                continue
            df = parse(raw, iso, name, year)
            print(f"  {iso} {year}: {len(df):,} HS6 lines "
                  f"nomen={sorted(df['nomen_code'].unique())}", flush=True)
            frames.append(df)
            time.sleep(3)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            f = OUT / f"clean_tariff_wits_hs6_{year}_g20.csv"
            out.to_csv(f, index=False)
            print(f"written {f} ({len(out):,} rows)", flush=True)
        else:
            print(f"year {year}: NO DATA AT ALL", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
