# -*- coding: utf-8 -*-
"""
Tariff scenario analysis on the real-data IO model (CHN/USA/ROW, 2022).

Scenario A: unilateral escalation -- USA raises the tariff on CHN
manufacturing by +0/+10/+25/+50/+100 pp on top of the 2022 baseline.
Scenario B: trade war -- both sides add +25 pp on each other's
manufacturing imports.

Reports welfare (real income %), the targeted bilateral flow, and the
change in US tariff revenue. Large steps use homotopy continuation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import get_real_data_io, solve_model


def tariff_revenue(sol, tau_mat):
    """Tariff revenue per region (scaled model units)."""
    x, p = sol["x"], sol["p"]
    return np.einsum("ris,ri,ris->s", tau_mat, p, x)


def apply_shocks(d, shocks, steps):
    res = solve_model(data=d, numeraire=("CHN", "labor"),
                      shock=shocks, homotopy_steps=steps)
    tau_mat = d["tau0"].copy()
    for o, i, s, rate in shocks:
        tau_mat[d["regions"].index(o), d["sectors"].index(i),
                d["regions"].index(s)] = rate
    return res, tau_mat


def load_bounds():
    """WTO bound ceilings per (importing region, sector); NaN if unknown."""
    df = pd.read_csv(HERE := Path(__file__).parent
                     / "data_real_io" / "bound_rates.csv")
    return df.set_index(["region", "sector"])["bound_rate"]


def cap_rate(target, importer, sector, bounds):
    """Cap a target tariff at the WTO bound ceiling (if known)."""
    b = bounds.get((importer, sector), np.nan)
    if np.isnan(b):
        return target, np.nan, False
    return min(target, b), b, target > b


def main() -> None:
    d = get_real_data_io()
    regions = d["regions"]
    o_i, k_i, s_i = (regions.index("CHN"), d["sectors"].index("manu"),
                     regions.index("USA"))
    base_us = float(d["tau0"][o_i, k_i, s_i])
    base_cn = float(d["tau0"][s_i, k_i, o_i])
    bounds = load_bounds()
    print(f"2022 baseline: US on CHN manu = {base_us:.2%}, "
          f"CHN on US manu = {base_cn:.2%}")
    print(f"WTO bound ceilings (manu): USA = {bounds[('USA','manu')]:.2%}, "
          f"CHN = {bounds[('CHN','manu')]:.2%}")

    # ---------- Scenario A: unilateral escalation, capped at bound ----------
    print("\n=== Scenario A: USA unilateral escalation on CHN manu "
          "(capped at WTO bound) ===")
    cache: dict[float, tuple] = {}
    rows = []
    for pp in [0.0, 0.10, 0.25, 0.50, 1.00]:
        target_raw = base_us + pp
        target, bound, capped = cap_rate(target_raw, "USA", "manu", bounds)
        if target not in cache:
            cache[target] = apply_shocks(
                d, [("CHN", "manu", "USA", target)],
                steps=max(1, int((target - base_us) / 0.005)))
        res, tau_mat = cache[target]
        b, c = res["benchmark"], res["counterfactual"]
        rr = c["real_income"] / b["real_income"]
        flow = (c["x"][o_i, k_i, s_i] / b["x"][o_i, k_i, s_i] - 1) * 100
        rev = (tariff_revenue(c, tau_mat)
               - tariff_revenue(b, d["tau0"]))   # change vs ACTUAL benchmark
        rows.append([pp * 100, target_raw * 100, target * 100,
                     "YES (illegal)" if capped else "no",
                     *[(rr[k] - 1) * 100 for k in range(3)],
                     flow, rev[s_i] * 1e4])
    A = pd.DataFrame(rows, columns=["added_pp", "target_pct_uncapped",
                                    "target_pct_capped", "exceeds_bound"]
                     + [f"welfare_{r}_pct" for r in regions]
                     + ["CHN_to_USA_manu_pct", "USA_tariff_revenue_chg_mUSD"])
    print(A.round(3).to_string(index=False))

    # ---------- Scenario B: trade war, both sides capped at bound ----------
    print("\n=== Scenario B: trade war, both sides +25pp "
          "(capped at bound) ===")
    t_us, b_us, cap_us = cap_rate(base_us + 0.25, "USA", "manu", bounds)
    t_cn, b_cn, cap_cn = cap_rate(base_cn + 0.25, "CHN", "manu", bounds)
    print(f"USA target: {base_us + 0.25:.2%} -> capped at {t_us:.2%} "
          f"(bound {b_us:.2%})" if cap_us else f"USA target: {t_us:.2%}")
    print(f"CHN target: {base_cn + 0.25:.2%} -> capped at {t_cn:.2%} "
          f"(bound {b_cn:.2%})" if cap_cn else f"CHN target: {t_cn:.2%}")
    res, tau_mat = apply_shocks(
        d, [("CHN", "manu", "USA", t_us), ("USA", "manu", "CHN", t_cn)],
        steps=5)
    b, c = res["benchmark"], res["counterfactual"]
    rr = c["real_income"] / b["real_income"]
    print("welfare % change:",
          {r: round((rr[k] - 1) * 100, 4) for k, r in enumerate(regions)})
    print("CHN->USA manu: "
          f"{(c['x'][o_i, k_i, s_i] / b['x'][o_i, k_i, s_i] - 1) * 100:.2f}%")
    print("USA->CHN manu: "
          f"{(c['x'][s_i, k_i, o_i] / b['x'][s_i, k_i, o_i] - 1) * 100:.2f}%")
    sec = pd.DataFrame(
        [(regions[o], d["sectors"][k],
          round((c["Y"][o, k] / b["Y"][o, k] - 1) * 100, 3))
         for o in range(3) for k in range(3)],
        columns=["region", "sector", "output_pct"])
    print(sec.to_string(index=False))


if __name__ == "__main__":
    main()
