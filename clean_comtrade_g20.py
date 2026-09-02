# -*- coding: utf-8 -*-
"""
Clean the 6 G20-expansion countries' Comtrade 2022 raw JSON
(CAN BRA ZAF RUS SAU AUS) into a long table with the standard schema
and filter rules.

RUS did not report to Comtrade for 2022 (sanctions); its export shares
are constructed from partners' import records (mirror) downstream.

Output: 整理/02_清洗/clean_comtrade_sitc_2022_g20.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path(r"D:\数据\CGE公开数据\09_UN_Comtrade")
OUT = Path(r"D:\数据\整理\02_清洗") / "clean_comtrade_sitc_2022_g20.csv"
REF = Path(r"D:\数据\整理\03_映射表\ref_country_v1.csv")

ISO = {124: "CAN", 76: "BRA", 710: "ZAF", 643: "RUS", 682: "SAU",
       36: "AUS"}

COLS = ["reporter_code", "reporter_iso3", "partner_code", "partner_iso3",
        "flow_code", "cmd_code", "cl_code", "period", "primary_value",
        "qty", "qty_unit_code", "net_wgt", "gross_wgt", "fob_value",
        "cif_value"]


def main() -> None:
    ref = pd.read_csv(REF, dtype={"m49": str})
    m49_iso = {}
    for r in ref.itertuples():
        if isinstance(r.m49, str) and r.m49.strip().isdigit():
            m49_iso[int(r.m49)] = r.iso3
    m49_iso[0] = "WLD"
    m49_iso[97] = "EU27"

    rows = []
    for code, iso in ISO.items():
        for cmd in ("TOTAL", "AG2"):
            f = SRC / f"COMTRADE-CA-{code:03d}-2022-S4-{cmd}.json"
            if not f.exists():
                print(f"MISSING: {f.name}")
                continue
            j = json.load(open(f, encoding="utf-8"))
            data = j["data"] if isinstance(j, dict) else j
            kept = filtered = 0
            for rec in data:
                if (rec.get("customsCode") not in ("C00", None)
                        or rec.get("motCode") not in (0, None)
                        or rec.get("partner2Code") not in (0, None)
                        or rec.get("flowCode") not in ("X", "M", "RM")):
                    filtered += 1
                    continue
                rows.append((
                    rec.get("reporterCode"), iso,
                    rec.get("partnerCode"),
                    m49_iso.get(rec.get("partnerCode")),
                    rec.get("flowCode"), rec.get("cmdCode"),
                    rec.get("classificationCode") or "S4",
                    rec.get("period"), rec.get("primaryValue"),
                    rec.get("qty"), rec.get("qtyUnitCode"),
                    rec.get("netWgt"), rec.get("grossWgt"),
                    rec.get("fobvalue"), rec.get("cifvalue"),
                ))
                kept += 1
            print(f"{iso}/{cmd}: kept {kept:,} filtered {filtered:,}")
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(OUT, index=False)
    print(f"\nwritten {OUT} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
