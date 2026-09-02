# -*- coding: utf-8 -*-
"""
14-region x 12-sector real-data CGE demo (2022).

Regions: CHN USA JPN KOR EU27 VNM THA MYS IDN SGP PHL IND MEX ROW
Shock: USA +25pp on CHN ELE (electronics), homotopy continuation.

Runs the standard verification suite first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import (REGIONS_14X12, get_real_data_14x12,
                                     solve_model)

SHOCK_SECTOR = "ELE"
SHOCK_PP = 0.25
HOMOTOPY_STEPS = 8


def verify(d, res, res2) -> None:
    b, c = res["benchmark"], res["counterfactual"]
    b2, c2 = res2["benchmark"], res2["counterfactual"]
    # Replication metrics: (a) share-weighted (economically meaningful),
    # (b) raw max (numerical noise detector, mostly epsilon cells).
    x_w = d["x0"] / d["x0"].sum()
    x_dev_w = float((np.abs(b["x"] / d["x0"] - 1) * x_w).sum())
    y_w = d["Y0"] / d["Y0"].sum()
    y_dev_w = float((np.abs(b["Y"] / d["Y0"] - 1) * y_w).sum())
    x_dev_max = np.abs(b["x"] / d["x0"] - 1).max()
    y_dev_max = np.abs(b["Y"] / d["Y0"] - 1).max()
    rep_price = max(np.abs(b["p"] - 1).max(), np.abs(b["w"] - 1).max(),
                    np.abs(b["I"] / d["I0"] - 1).max())
    print(f"[1] benchmark replication:")
    print(f"    weighted dev:  x {x_dev_w:.3e}  Y {y_dev_w:.3e}")
    print(f"    max dev:       x {x_dev_max:.3e}  Y {y_dev_max:.3e}  "
          f"prices/income {rep_price:.3e}")
    print(f"    goods resid:   {np.abs(b['goods_log_resid']).max():.3e}")
    assert x_dev_w < 1e-5 and y_dev_w < 1e-5 and rep_price < 1e-4
    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[2] goods-market clearing under shock: max |log resid| = {gm:.3e}")
    assert gm < 1e-6
    floor_x = d["x0"] > 2.0 * 1e-4
    floor_y = d["Y0"] > 100.0 * 1e-4
    dev = max(
        np.abs((c["x"] / c2["x"] - 1)[floor_x]).max(),
        np.abs((c["Y"] / c2["Y"] - 1)[floor_y]).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[3] numeraire invariance: max dev = {dev:.3e}")
    assert dev < 1e-4
    print("All verification checks passed.\n")


def main() -> None:
    d = get_real_data_14x12()
    regions = d["regions"]
    o_i = regions.index("CHN")
    k_i = d["sectors"].index(SHOCK_SECTOR)
    s_i = regions.index("USA")
    target = float(d["tau0"][o_i, k_i, s_i]) + SHOCK_PP
    shock = ("CHN", SHOCK_SECTOR, "USA", target)
    print(f"shock: USA tariff on CHN {SHOCK_SECTOR}: "
          f"{d['tau0'][o_i, k_i, s_i]:.2%} -> {target:.2%} (+{SHOCK_PP:.0%} pp)\n")

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                      homotopy_steps=HOMOTOPY_STEPS)
    res2 = solve_model(data=d, numeraire=("USA", "labor"), shock=shock,
                       homotopy_steps=HOMOTOPY_STEPS)
    verify(d, res, res2)

    b, c = res["benchmark"], res["counterfactual"]
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * 1e4 * (rr - 1),
    })
    print("welfare (household real consumption) and EV:")
    print(wdf.round(3).to_string(index=False))

    print(f"\nCHN {SHOCK_SECTOR} exports by destination, % change:")
    for s2 in range(len(regions)):
        if s2 == o_i:
            continue
        chg = (c["x"][o_i, k_i, s2] / b["x"][o_i, k_i, s2] - 1) * 100
        print(f"  CHN -> {regions[s2]:<5} {chg:+.2f}%")

    sec = pd.DataFrame(
        [(regions[o], d["sectors"][k],
          (c["Y"][o, k] / b["Y"][o, k] - 1) * 100)
         for o in range(len(regions)) for k in range(len(d["sectors"]))],
        columns=["region", "sector", "output_pct"])
    print("\nsector output % change:")
    print(sec.pivot(index="sector", columns="region",
                    values="output_pct").round(3).to_string())


if __name__ == "__main__":
    main()
