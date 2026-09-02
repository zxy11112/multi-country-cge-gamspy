# -*- coding: utf-8 -*-
"""
Download UN Comtrade 2022 (SITC Rev.4, TOTAL + AG2) for the 6 G20-style
expansion reporters: CAN BRA ZAF RUS SAU AUS.
Same pattern as download_comtrade_expansion.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API_KEY = "f0faf8e8bc36428c91685eaf69a320c5"
OUT_DIR = Path(r"D:\数据\CGE公开数据\09_UN_Comtrade")
BASE = "https://comtradeapi.un.org/data/v1/get/C/A/S4"

# M49/Comtrade reporter codes
REPORTERS = {"CAN": 124, "BRA": 76, "ZAF": 710, "RUS": 643,
             "SAU": 682, "AUS": 36}

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
                n = len(j.get("data", []))
                f.write_text(json.dumps(j), encoding="utf-8")
                flag = " ** HIT CAP, needs split **" if n >= 100000 else ""
                print(f"{iso}/{cmd}: ok, records={n}{flag}")
            else:
                print(f"{iso}/{cmd}: http {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"{iso}/{cmd}: ERROR {e}")
        time.sleep(3)
print("DONE")
