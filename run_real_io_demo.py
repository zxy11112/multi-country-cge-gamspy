# -*- coding: utf-8 -*-
"""
Real-data demo WITH intermediate inputs: CHN / USA / ROW x
{prim, manu, serv}, benchmark year 2022.

Experiment: USA imposes an additional 25% ad valorem tariff on
manufacturing imports from CHN.

Also runs the standard GAMSPy verification suite on the real dataset:
benchmark replication, goods-market clearing, numeraire invariance.

Data: data_real_io/*.csv built by prepare_real_data_io.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import get_real_data_io, solve_model

SHOCK = ("CHN", "manu", "USA", 0.25)   # origin, sector, destination, rate


def verify(d, res, res2) -> None:
    b, c = res["benchmark"], res["counterfactual"]
    b2, c2 = res2["benchmark"], res2["counterfactual"]

    rep = max(
        np.abs(b["p"] - 1).max(),
        np.abs(b["w"] - 1).max(),
        np.abs(b["Y"] / d["Y0"] - 1).max(),
        np.abs(b["I"] / d["I0"] - 1).max(),
        np.abs(b["x"] / d["x0"] - 1).max(),
        np.abs(b["goods_log_resid"]).max(),
    )
    print(f"[1] benchmark replication: max deviation = {rep:.3e}")
    assert rep < 1e-6

    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[2] goods-market clearing under shock: max |log resid| = {gm:.3e}")
    assert gm < 1e-6

    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[3] numeraire invariance (real IO data): max dev = {dev:.3e}")
    assert dev < 1e-6
    print("All verification checks passed.\n")


def main() -> None:
    d = get_real_data_io()
    regions, sectors = d["regions"], d["sectors"]
    R, S = len(regions), len(sectors)

    # shock: baseline tariff + 25 percentage points on CHN->USA manu
    o_i, k_i, s_i = (regions.index(SHOCK[0]), sectors.index(SHOCK[1]),
                     regions.index(SHOCK[2]))
    target = float(d["tau0"][o_i, k_i, s_i]) + SHOCK[3]
    shock = (SHOCK[0], SHOCK[1], SHOCK[2], target)
    print(f"shock: {SHOCK[2]} raises tariff on {SHOCK[1]} from {SHOCK[0]} "
          f"from {d['tau0'][o_i, k_i, s_i]:.2%} to {target:.2%} "
          f"(+{SHOCK[3]:.0%} pp)\n")

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                      homotopy_steps=5)
    res2 = solve_model(data=d, numeraire=("USA", "labor"), shock=shock,
                       homotopy_steps=5)
    verify(d, res, res2)

    b, c = res["benchmark"], res["counterfactual"]

    # ---------- bilateral flow changes (all sectors) ----------
    print(f"=== shock: USA +{SHOCK[3]:.0%} tariff on {SHOCK[1]} imports "
          f"from {SHOCK[0]} ===\n")
    rows = []
    for o in range(R):
        for k in range(S):
            for s2 in range(R):
                if o == s2:
                    continue
                rows.append((regions[o], sectors[k], regions[s2],
                             (c["x"][o, k, s2] / b["x"][o, k, s2] - 1) * 100))
    chg = pd.DataFrame(rows, columns=["origin", "sector", "destination",
                                      "flow_pct_change"])
    print("international bilateral flows, % change:")
    print(chg.round(2).to_string(index=False))

    # USA manufacturing demand by origin
    jU, kM = regions.index("USA"), sectors.index("manu")
    div = pd.DataFrame({
        "origin": regions,
        "benchmark_mUSD": b["x"][:, kM, jU] * 1e4,
        "counterfactual_mUSD": c["x"][:, kM, jU] * 1e4,
        "pct_change": (c["x"][:, kM, jU] / b["x"][:, kM, jU] - 1) * 100,
    })
    print("\nUSA manufacturing absorption by origin:")
    print(div.round(2).to_string(index=False))

    # ---------- welfare ----------
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * 1e4 * (rr - 1),   # data are scaled by 1e-4
    })
    print("\nwelfare (real income) and equivalent variation:")
    print(wdf.round(3).to_string(index=False))

    # ---------- sector output ----------
    sec = pd.DataFrame(
        [(regions[o], sectors[k],
          (c["Y"][o, k] / b["Y"][o, k] - 1) * 100,
          (c["p"][o, k] / b["p"][o, k] - 1) * 100)
         for o in range(R) for k in range(S)],
        columns=["region", "sector", "output_pct", "producer_price_pct"])
    print("\nsector output / producer price changes:")
    print(sec.round(3).to_string(index=False))

    # ---------- factor prices ----------
    fac = pd.DataFrame(
        [(regions[o], d["factors"][k],
          (c["w"][o, k] / b["w"][o, k] - 1) * 100)
         for o in range(R) for k in range(2)],
        columns=["region", "factor", "price_pct"])
    print("\nfactor price changes (numeraire: CHN labor):")
    print(fac.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
