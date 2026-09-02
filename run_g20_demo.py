# -*- coding: utf-8 -*-
"""G20-flavor 14-region x 12-sector demo: verify + US +25pp on CHN ELE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import (REGIONS_G20, get_real_data_g20,
                                     solve_model)

SHOCK_SECTOR = "ELE"
SHOCK_PP = 0.25


def verify(d, res, res2) -> None:
    b, c = res["benchmark"], res["counterfactual"]
    b2, c2 = res2["benchmark"], res2["counterfactual"]
    x_w = d["x0"] / d["x0"].sum()
    y_w = d["Y0"] / d["Y0"].sum()
    print("[1] benchmark replication:")
    print(f"    weighted dev:  x "
          f"{float((np.abs(b['x']/d['x0']-1)*x_w).sum()):.3e}  Y "
          f"{float((np.abs(b['Y']/d['Y0']-1)*y_w).sum()):.3e}")
    print(f"    max dev:       x {np.abs(b['x']/d['x0']-1).max():.3e}  "
          f"prices {np.abs(b['p']-1).max():.3e}")
    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[2] goods-market clearing under shock: {gm:.3e}")
    assert gm < 1e-6
    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[3] numeraire invariance: max dev = {dev:.3e}")
    assert dev < 1e-3
    print("All verification checks passed.\n")


def main() -> None:
    d = get_real_data_g20()
    regions = d["regions"]
    o_i = regions.index("CHN")
    k_i = d["sectors"].index(SHOCK_SECTOR)
    s_i = regions.index("USA")
    target = float(d["tau0"][o_i, k_i, s_i]) + SHOCK_PP
    shock = ("CHN", SHOCK_SECTOR, "USA", target)
    print(f"shock: USA tariff on CHN {SHOCK_SECTOR}: "
          f"{d['tau0'][o_i, k_i, s_i]:.2%} -> {target:.2%}\n")

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                      homotopy_steps=8)
    res2 = solve_model(data=d, numeraire=("USA", "labor"), shock=shock,
                       homotopy_steps=8)
    verify(d, res, res2)

    for label, r in [("numeraire=CHN", res), ("numeraire=USA", res2)]:
        t = r["timing"]
        print(f"\n[{label}] timing:")
        print(f"  total:                 {t['total_seconds']:.2f}s")
        print(f"  model build:           {t['build_seconds']:.2f}s")
        print(f"  benchmark solve:       {t['benchmark_solve_seconds']:.2f}s")
        print(f"  counterfactual solve:  {t['counterfactual_solve_seconds']:.2f}s")
        print(f"  homotopy steps:        {t['homotopy_steps']}")
        print(f"  avg step:              {sum(t['per_step_seconds'])/max(1,len(t['per_step_seconds'])):.2f}s")

    b, c = res["benchmark"], res["counterfactual"]
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * 1e4 * (rr - 1),
    })
    print("welfare and EV:")
    print(wdf.round(3).to_string(index=False))

    print(f"\nCHN {SHOCK_SECTOR} exports by destination, % change:")
    for s2 in range(len(regions)):
        if s2 == o_i:
            continue
        chg = (c["x"][o_i, k_i, s2] / b["x"][o_i, k_i, s2] - 1) * 100
        print(f"  CHN -> {regions[s2]:<5} {chg:+.2f}%")


if __name__ == "__main__":
    main()
