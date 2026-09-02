# -*- coding: utf-8 -*-
"""
GAMSPy-only verification of the real-data demo (CHN/USA/ROW, 2022):

1. Benchmark replication: solving with benchmark tariffs (all zero)
   must reproduce the input data exactly.
2. Numeraire invariance: solving the same tariff experiment with
   numeraire CHN:labor and USA:labor must give identical real
   allocations (quantities, outputs, real-income ratios).
3. Goods-market clearing: post-shock log(Y / sum_s x) residuals,
   including every market.
"""

from __future__ import annotations

import numpy as np

from multicountry_cge_gamspy import get_real_data, solve_model

SHOCK = ("CHN", "manu", "USA", 0.25)


def main() -> None:
    d = get_real_data()

    # ---- 1. benchmark replication ----
    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=SHOCK)
    b, c = res["benchmark"], res["counterfactual"]
    rep = max(
        np.abs(b["p"] - 1).max(),
        np.abs(b["w"] - 1).max(),
        np.abs(b["Y"] / d["Y0"] - 1).max(),
        np.abs(b["I"] / d["I0"] - 1).max(),
        np.abs(b["x"] / d["x0"] - 1).max(),
        np.abs(b["goods_log_resid"]).max(),
    )
    print(f"[1] benchmark replication: max deviation = {rep:.3e}")
    assert rep < 1e-6, "benchmark replication failed"

    # ---- 3. market clearing under the shock (all 6 markets) ----
    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[3] goods-market clearing under shock: max |log resid| = {gm:.3e}")
    assert gm < 1e-6, "goods markets do not clear"

    # ---- 2. numeraire invariance on the real dataset ----
    res2 = solve_model(data=d, numeraire=("USA", "labor"), shock=SHOCK)
    b2, c2 = res2["benchmark"], res2["counterfactual"]
    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[2] numeraire invariance (real data): max dev = {dev:.3e}")
    assert dev < 1e-6, "numeraire invariance failed"

    print("\nAll GAMSPy verification checks passed.")


if __name__ == "__main__":
    main()
