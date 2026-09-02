# -*- coding: utf-8 -*-
"""
6-region x 12-sector real-data CGE demo (2022, intermediates,
nested Armington, government + savings/investment).

Regions : CHN USA JPN KOR EU27 ROW
Sectors : AGF MIN ENR CHM TXL WDP NMM MAC ELE VEH OTM SRV

Shock: USA raises its tariff on CHN electronics (ELE) by +25 pp
on top of the 2022 baseline, solved with homotopy continuation.

Runs the standard verification suite first: benchmark replication,
goods-market clearing, numeraire invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import (REGIONS_6X12, SECTORS_6X12,
                                     get_real_data_6x12, solve_model)

SHOCK_SECTOR = "ELE"
SHOCK_PP = 0.25
HOMOTOPY_STEPS = 5


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
    # 6x12: PATH stops at ~1e-5 on prices and ~4.5e-5 on the smallest
    # (floored) flows of the smallest region; equation residuals remain
    # at machine precision. Prices/outputs/incomes deviate <= 1.2e-5.
    assert rep < 1e-4
    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[2] goods-market clearing under shock: max |log resid| = {gm:.3e}")
    assert gm < 1e-6
    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[3] numeraire invariance: max dev = {dev:.3e}")
    # same PATH-noise scale as [1]; structural violations (like the
    # B-denomination bug) show up at 1e-2, two orders above this.
    assert dev < 1e-4
    print("All verification checks passed.\n")


def main() -> None:
    d = get_real_data_6x12()
    regions, sectors = d["regions"], d["sectors"]
    R, S = len(regions), len(sectors)
    o_i, k_i, s_i = (regions.index("CHN"), sectors.index(SHOCK_SECTOR),
                     regions.index("USA"))
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

    # ---------- welfare ----------
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * 1e4 * (rr - 1),
    })
    print("welfare (household real consumption) and EV:")
    print(wdf.round(3).to_string(index=False))

    # ---------- CHN ELE exports by destination ----------
    print(f"\nCHN {SHOCK_SECTOR} exports by destination, % change:")
    for s2 in range(R):
        if s2 == 0:
            continue
        chg = (c["x"][o_i, k_i, s2] / b["x"][o_i, k_i, s2] - 1) * 100
        print(f"  CHN -> {regions[s2]:<5} {chg:+.2f}%")

    # ---------- USA ELE absorption by origin ----------
    div = pd.DataFrame({
        "origin": regions,
        "benchmark_bnUSD": b["x"][:, k_i, s_i] * 1e4 / 1e3,
        "counterfactual_bnUSD": c["x"][:, k_i, s_i] * 1e4 / 1e3,
        "pct_change": (c["x"][:, k_i, s_i] / b["x"][:, k_i, s_i] - 1) * 100,
    })
    print(f"\nUSA {SHOCK_SECTOR} absorption by origin (bn USD):")
    print(div.round(2).to_string(index=False))

    # ---------- sector outputs for CHN and USA ----------
    sec = pd.DataFrame(
        [(regions[o], sectors[k],
          (c["Y"][o, k] / b["Y"][o, k] - 1) * 100)
         for o in range(R) for k in range(S)],
        columns=["region", "sector", "output_pct"])
    print("\nsector output % change:")
    print(sec.pivot(index="sector", columns="region",
                    values="output_pct").round(3).to_string())


if __name__ == "__main__":
    main()
