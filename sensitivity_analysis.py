# -*- coding: utf-8 -*-
"""
Sensitivity analysis of the Armington elasticities (sigma) for the
real-data IO demo (CHN/USA/ROW, 2022, intermediate inputs).

The shock is the same as run_real_io_demo.py (USA tariff on CHN
manufacturing: baseline + 25 pp). The whole sigma vector is scaled by
factors from 0.5 to 1.5; welfare and the key trade flow are reported
for each factor. Edit data_real_io/elasticities.csv for the central
values; this script only multiplies them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import get_real_data_io, solve_model

# NOTE: factors must keep every sector's sigma > 1 (CES requires it).
# serv has sigma = 2.0, so the lowest factor is 0.6 (0.6 x 2.0 = 1.2).
FACTORS = [0.6, 0.75, 1.0, 1.25, 1.5]
SHOCK_PP = 0.25
HOMOTOPY_STEPS = 5


def main() -> None:
    d0 = get_real_data_io()
    regions = d0["regions"]
    o_i = regions.index("CHN")
    k_i = d0["sectors"].index("manu")
    s_i = regions.index("USA")
    target = float(d0["tau0"][o_i, k_i, s_i]) + SHOCK_PP
    shock = ("CHN", "manu", "USA", target)

    rows = []
    for f in FACTORS:
        d = dict(d0)
        d["sigma"] = d0["sigma"] * f
        res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                          homotopy_steps=HOMOTOPY_STEPS)
        b, c = res["benchmark"], res["counterfactual"]
        rr = c["real_income"] / b["real_income"]
        flow = c["x"][o_i, k_i, s_i] / b["x"][o_i, k_i, s_i] - 1
        rows.append([f,
                     *[(rr[k] - 1) * 100 for k in range(len(regions))],
                     flow * 100])
        print(f"sigma x {f:.2f} solved.")

    out = pd.DataFrame(
        rows, columns=["sigma_factor"]
        + [f"welfare_{r}_pct" for r in regions]
        + ["CHN_to_USA_manu_pct"])
    print("\n=== sigma sensitivity (USA +25pp on CHN manu) ===")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
