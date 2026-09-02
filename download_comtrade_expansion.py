# -*- coding: utf-8 -*-
"""
Download UN Comtrade 2022 (SITC Rev.4, TOTAL + AG2) for the 8 expansion
reporters: VNM THA MYS IDN SGP PHL IND MEX.

Pattern follows the existing download_un_comtrade_s4_2010_2023.js:
  GET https://comtradeapi.un.org/data/v1/get/C/A/S4
      ?reportercode={m49}&period=2022&cmdCode={TOTAL|AG2}&maxRecords=100000
  header: Ocp-Apim-Subscription-Key

Output naming matches the existing raw files:
  COMTRADE-CA-{code:03d}-2022-S4-{TOTAL|AG2}.json
in CGE公开数据/09_UN_Comtrade/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API_KEY = "f0faf8e8bc36428c91685eaf69a320c5"   # primary key (user-provided)
OUT_DIR = Path(r"D:\数据\CGE公开数据\09_UN_Comtrade")
BASE = "https://comtradeapi.un.org/data/v1/get/C/A/S4"

REPORTERS = {"VNM": 704, "THA": 764, "MYS": 458, "IDN": 360,
             "SGP": 702, "PHL": 608, "IND": 699, "MEX": 484}

for iso, code in REPORTERS.items():
    for cmd in ("TOTAL", "AG2"):
        name = f"COMTRADE-CA-{code:03d}-2022-S4-{cmd}.json"
        f = OUT_DIR / name
        if f.exists() and f.stat().st_size > 1000:
            print(f"{iso}/{cmd}: skip (exists)")
            continue
        params = {"reportercode": code, "period": 2022, "cmdCode": cmd,
                  "maxRecords": 100000}
        try:
            r = requests.get(BASE, params=params,
                             headers={"Ocp-Apim-Subscription-Key": API_KEY,
                                      "Accept": "application/json"},
                             timeout=300)
            if r.status_code == 200:
                j = r.json()
                n = j.get("count", len(j.get("data", [])))
                f.write_text(json.dumps(j), encoding="utf-8")
                print(f"{iso}/{cmd}: ok, count={n}")
            else:
                print(f"{iso}/{cmd}: http {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"{iso}/{cmd}: ERROR {e}")
        time.sleep(3)
print("DONE")
