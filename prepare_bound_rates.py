# -*- coding: utf-8 -*-
"""
Compute WTO bound tariff ceilings per model sector for CHN and USA from
the cleaned CTS data (clean_tariff_cts_bound_v1.csv).

Output: data_real_io/bound_rates.csv  (region, sector, bound_rate)

Method: simple average of ad valorem bound rates over national tariff
lines, aggregated by HS chapter (01-27 -> prim, 28-97 -> manu).
Lines with specific/compound duties are excluded (not converted), so
ceilings are understated for agriculture. HS versions differ by member
(CHN HS2002, USA HS2007R5) but the 2-digit chapter split is stable.

Caveat: the legal constraint binds per tariff line; a sector-average
ceiling is an approximation for the aggregated model sector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\数据")
CTS = ROOT / "整理" / "02_清洗" / "clean_tariff_cts_bound_v1.csv"
OUT = ROOT / "simple_multicountry_cge" / "data_real_io"


def main() -> None:
    df = pd.read_csv(CTS, usecols=["reporter_iso3", "hs_code",
                                   "duty_value", "duty_unit"],
                     dtype={"hs_code": str})
    df = df[(df["reporter_iso3"].isin(["CHN", "USA"]))
            & (df["duty_unit"] == "ad_valorem")]
    # keep pure ad valorem entries only ("x%" or "Free"); drop text cells
    dv = df["duty_value"].str.strip()
    is_pct = dv.str.match(r"^\d+(\.\d+)?%$")
    is_free = dv.str.lower().isin(["free", "0"])
    df = df[is_pct | is_free].copy()
    df["rate"] = np.where(is_free[df.index], 0.0,
                          dv[df.index].str.rstrip("%").astype(float)) / 100.0
    df["chapter"] = df["hs_code"].str[:2].astype(int)
    df["sector"] = np.where(df["chapter"] <= 27, "prim", "manu")
    g = df.groupby(["reporter_iso3", "sector"])["rate"].mean()

    rows = [(reg, sec, float(g[(reg, sec)]))
            for reg in ["CHN", "USA"] for sec in ["prim", "manu"]]
    rows += [("ROW", "prim", np.nan), ("ROW", "manu", np.nan)]  # no data
    out = pd.DataFrame(rows, columns=["region", "sector", "bound_rate"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "bound_rates.csv", index=False)
    print("bound rates (simple average of ad valorem lines):")
    print(out.round(4).to_string(index=False))
    print(f"\nwritten to {OUT / 'bound_rates.csv'}")


if __name__ == "__main__":
    main()
