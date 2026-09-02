# -*- coding: utf-8 -*-
"""
Clean the 8 expansion countries' Comtrade 2022 raw JSON into a long table
with the same schema and filter rules as clean_comtrade_sitc_1992-2023_v1.1.csv:
  customsCode == C00, motCode == 0, partner2Code == 0,
  flowCode in {X, M, RM}

Input : CGE公开数据/09_UN_Comtrade/COMTRADE-CA-{code}-2022-S4-{TOTAL|AG2}.json
        (compact JSON written by download_comtrade_expansion.py)
Output: 整理/02_清洗/clean_comtrade_sitc_2022_expansion.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path(r"D:\数据\CGE公开数据\09_UN_Comtrade")
OUT = Path(r"D:\数据\整理\02_清洗") / "clean_comtrade_sitc_2022_expansion.csv"

ISO = {704: "VNM", 764: "THA", 458: "MYS", 360: "IDN",
       702: "SGP", 608: "PHL", 699: "IND", 484: "MEX"}

COLS = ["reporter_code", "reporter_iso3", "partner_code", "partner_iso3",
        "flow_code", "cmd_code", "cl_code", "period", "primary_value",
        "qty", "qty_unit_code", "net_wgt", "gross_wgt", "fob_value",
        "cif_value"]

# partner ISO3 mapping comes from the reference table (M49 -> iso3)
REF = Path(r"D:\数据\整理\03_映射表\ref_country_v1.csv")


def main() -> None:
    ref = pd.read_csv(REF, dtype={"m49": str})
    m49_iso = {}
    for r in ref.itertuples():
        if isinstance(r.m49, str) and r.m49.strip().isdigit():
            m49_iso[int(r.m49)] = r.iso3
    m49_iso[0] = "WLD"
    m49_iso[97] = "EU27"

    rows = []
    for iso, code in [(v, k) for k, v in ISO.items()]:
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
                    m49_iso.get(rec.get("partnerCode"), None),
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
