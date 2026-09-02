# -*- coding: utf-8 -*-
"""
Re-download TRUNCATED WITS HS6 XMLs chapter by chapter (HS 2-digit
product codes), then merge into the whole-country XML file.

The WITS server is slow; whole-country downloads get cut by the 900s
timeout. Per-chapter requests are small and fast.

Usage: python download_wits_expansion_retry.py  (auto-detects truncated
files in the raw folder and re-downloads them chapter-wise)
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

RAW = Path(r"D:\数据\CGE公开数据\10_WorldBank_WITS\HS6关税表_raw")

COUNTRIES = {"704": "越南", "764": "泰国", "458": "马来西亚",
             "360": "印尼", "702": "新加坡", "608": "菲律宾",
             "356": "印度", "484": "墨西哥"}

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


def is_truncated(path: Path) -> bool:
    try:
        ET.parse(path)
        return False
    except ET.ParseError:
        return True


def download_chapter(code: str, chapter: str) -> str:
    url = ("https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
           f"DF_WITS_Tariff_TRAINS/.{code}.000.{chapter}.reported/"
           "?startperiod=2022&endperiod=2022")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.text


def extract_series(xml_text: str) -> list[str]:
    """Pull <generic:Series>...</generic:Series> blocks as raw strings."""
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


def main() -> None:
    for code, name in COUNTRIES.items():
        f = RAW / f"WITS_HS6_{int(code):03d}_2022.xml"
        if f.exists() and not is_truncated(f):
            print(f"{name}: complete, skip")
            continue
        print(f"{name}: missing or truncated -> re-download per chapter")
        series_all = []
        for i, ch in enumerate(CHAPTERS):
            try:
                txt = download_chapter(code, ch)
                series_all.extend(extract_series(txt))
            except Exception as e:
                print(f"  {name} ch{ch}: {e}")
            time.sleep(1)
            if i % 20 == 0:
                print(f"  {name}: {i}/97 chapters, {len(series_all)} series")
        f.write_text(HDR + "".join(series_all) + FTR, encoding="utf-8")
        ok = not is_truncated(f)
        print(f"  {name}: {len(series_all)} series, parses OK: {ok}")
    print("DONE")


if __name__ == "__main__":
    main()
