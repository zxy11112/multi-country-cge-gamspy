# -*- coding: utf-8 -*-
"""
Real-data demo: CHN / USA / ROW x {prim, manu}, benchmark year 2022.

Experiment: USA imposes an additional 25% ad valorem tariff on
manufacturing imports from CHN (301-style shock), solved with the
GAMSPy MCP model in multicountry_cge_gamspy.py.

Data: data_real/*.csv built by prepare_real_data.py
(OECD IOTs value added + UN Comtrade bilateral goods trade).

Units: million USD. Reported changes are model simulations on a
demo-grade dataset (see prepare_real_data.py docstring for the
value-added/gross-flow caveat and the SITC<->ISIC approximation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import get_real_data, solve_model

SHOCK = ("CHN", "manu", "USA", 0.25)   # origin, sector, destination, rate


def main() -> None:
    d = get_real_data()
    regions, sectors = d["regions"], d["sectors"]
    R, S = len(regions), len(sectors)

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=SHOCK)
    b, c = res["benchmark"], res["counterfactual"]

    # ---------- benchmark replication ----------
    rep = max(
        np.abs(b["p"] - 1).max(),
        np.abs(b["w"] - 1).max(),
        np.abs(b["Y"] / d["Y0"] - 1).max(),
        np.abs(b["I"] / d["I0"] - 1).max(),
        np.abs(b["x"] / d["x0"] - 1).max(),
    )
    print("=== benchmark replication (real data, 2022) ===")
    print(f"max deviation: {rep:.3e}")
    assert rep < 1e-6, "benchmark replication failed"

    # ---------- bilateral flow changes ----------
    print(f"\n=== shock: USA +{SHOCK[3]:.0%} tariff on {SHOCK[1]} imports "
          f"from {SHOCK[0]} ===")
    rows = []
    for o in range(R):
        for s in range(R):
            if o == s:
                continue
            k = sectors.index("manu")
            rows.append((regions[o], regions[s],
                         c["x"][o, k, s] / b["x"][o, k, s] - 1))
    chg = pd.DataFrame(rows, columns=["origin", "destination",
                                      "manu_flow_pct_change"])
    chg["manu_flow_pct_change"] = (chg["manu_flow_pct_change"] * 100).round(2)
    print("\nmanufacturing bilateral flows, % change:")
    print(chg.to_string(index=False))

    # USA manufacturing demand by origin (trade diversion view)
    j = regions.index("USA")
    k = sectors.index("manu")
    div = pd.DataFrame({
        "origin": regions,
        "benchmark_mUSD": b["x"][:, k, j],
        "counterfactual_mUSD": c["x"][:, k, j],
        "pct_change": (c["x"][:, k, j] / b["x"][:, k, j] - 1) * 100,
    })
    print("\nUSA manufacturing absorption by origin:")
    print(div.round(2).to_string(index=False))

    # ---------- welfare ----------
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * (rr - 1),
    })
    print("\nwelfare (real income) and equivalent variation:")
    print(wdf[["region", "welfare_pct", "EV_mUSD"]].round(3).to_string(index=False))

    # ---------- sector output and factor prices ----------
    sec = pd.DataFrame(
        [(regions[o], sectors[k],
          (c["Y"][o, k] / b["Y"][o, k] - 1) * 100,
          (c["p"][o, k] / b["p"][o, k] - 1) * 100)
         for o in range(R) for k in range(S)],
        columns=["region", "sector", "output_pct", "producer_price_pct"])
    print("\nsector output / producer price changes:")
    print(sec.round(3).to_string(index=False))

    print("\nmax |goods-market log residual| (counterfactual): "
          f"{np.abs(c['goods_log_resid']).max():.3e}")


if __name__ == "__main__":
    main()
