# -*- coding: utf-8 -*-
"""
Literature comparison: full US-China trade war scenario on the
6x12 model (2022 benchmark).

Scenario (approximating the 2018-19 trade war):
  - USA +25pp on ALL goods-sector imports from CHN
  - CHN +25pp on ALL goods-sector imports from USA (retaliation)
Solved with homotopy continuation.

Reference points from the literature (2018-19 actual tariffs):
  - Fajgelbaum et al. (2020, QJE): US welfare -0.04% of GDP
  - Grossman-Helpman-Redding (2023): ~-0.12% of GDP (US)
  - Chang et al. (2021, replicating FKPG): China -0.29% of GDP
  - Ma (2024): China -0.29% of GDP
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import (REGIONS_6X12, SECTORS_6X12,
                                     get_real_data_6x12, solve_model)


def main() -> None:
    d = get_real_data_6x12()
    regions = d["regions"]
    goods = [s for s in d["sectors"] if s != "SRV"]

    # both directions, all goods sectors, +25pp on top of baseline
    shocks = []
    for sec in goods:
        for o, s_ in [("CHN", "USA"), ("USA", "CHN")]:
            o_i, k_i, s_i = (regions.index(o), d["sectors"].index(sec),
                             regions.index(s_))
            target = float(d["tau0"][o_i, k_i, s_i]) + 0.25
            shocks.append((o, sec, s_, target))

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shocks,
                      homotopy_steps=8)
    b, c = res["benchmark"], res["counterfactual"]

    # ---- welfare ----
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
    })
    print("=== full trade war (+25pp both directions, all goods) ===")
    print("welfare % change (household real consumption):")
    print(wdf.round(3).to_string(index=False))

    # ---- GDP-relative (for literature comparison) ----
    gdp = d["I0"] * 1e4           # household income as model GDP proxy
    print("\nwelfare as % of model income base:")
    for k, r in enumerate(regions):
        ev = gdp[k] * (rr[k] - 1)
        print(f"  {r}: {(rr[k]-1)*100:+.3f}%  (EV {ev/1e3:+,.1f} bn USD)")

    # ---- bilateral trade: CHN<->USA goods ----
    o_c, s_u = regions.index("CHN"), regions.index("USA")
    chn_usa = sum(c["x"][o_c, k, s_u] for k in range(len(d["sectors"]) - 1))
    chn_usa_b = sum(b["x"][o_c, k, s_u] for k in range(len(d["sectors"]) - 1))
    usa_chn = sum(c["x"][s_u, k, o_c] for k in range(len(d["sectors"]) - 1))
    usa_chn_b = sum(b["x"][s_u, k, o_c] for k in range(len(d["sectors"]) - 1))
    print(f"\nCHN->USA goods: {(chn_usa/chn_usa_b-1)*100:+.1f}%")
    print(f"USA->CHN goods: {(usa_chn/usa_chn_b-1)*100:+.1f}%")

    print("\n=== literature reference points (2018-19 tariffs) ===")
    print("  Fajgelbaum et al. (2020 QJE): US -0.04% of GDP")
    print("  Grossman-Helpman-Redding (2023): US ~-0.12% of GDP")
    print("  Chang et al. (2021) / Ma (2024): China -0.29% of GDP")


if __name__ == "__main__":
    main()
