# -*- coding: utf-8 -*-
"""
Batch-download GFS SOO G1151 (customs & import duties, 2022, local
currency) for the 49 ROW economies via the IMF SDMX 2.1 API.

Saves one small CSV per economy to 12_IMF_GFS/明细_G1151_ROW/.
Existing files are skipped (idempotent re-runs).
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

OUT = Path(r"D:\数据\CGE公开数据\12_IMF_GFS") / "明细_G1151_ROW"
OUT.mkdir(parents=True, exist_ok=True)

ROW = ("AGO ARE ARG AUS BGD BLR BRA BRN CAN CHE CHL CIV CMR COD COL CRI "
       "EGY GBR HKG IDN IND ISL ISR JOR KAZ KHM LAO MAR MEX MMR MYS NGA "
       "NOR NZL PAK PER PHL RUS SAU SEN SGP STP THA TUN TUR TWN UKR VNM ZAF").split()

URL = ("https://api.imf.org/external/sdmx/2.1/data/GFS_SOO/"
       "{cc}.S13..G1151_T.XDC.A?startPeriod=2022&endPeriod=2022")

for cc in ROW:
    f = OUT / f"GFS_SOO_{cc}_G1151_XDC.csv"
    if f.exists() and f.stat().st_size > 500:
        print(f"{cc}: skip (exists)")
        continue
    try:
        r = requests.get(URL.format(cc=cc),
                         headers={"Accept": "text/csv"}, timeout=120)
        if r.status_code == 200 and "OBS_VALUE" in r.text:
            f.write_text(r.text, encoding="utf-8")
            print(f"{cc}: ok ({len(r.text)} bytes)")
        else:
            print(f"{cc}: http {r.status_code}")
    except Exception as e:
        print(f"{cc}: ERROR {e}")
    time.sleep(2)
print("DONE")
