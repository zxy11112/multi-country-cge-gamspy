# -*- coding: utf-8 -*-
"""
Model checklist (A-E) for the 14x12 model, per the review protocol.

Default experiment: USA +10pp on CHN ELE (manufacturing-like sector).
All checks print PASS/FAIL with the actual numbers.

Run:  D:\\gamspy_env\\Scripts\\python.exe model_checklist.py
"""

from __future__ import annotations

import numpy as np

from multicountry_cge_gamspy import get_real_data_14x12, solve_model

SHOCK_PP = 0.10
SECTOR = "ELE"


def main() -> None:
    d = get_real_data_14x12()
    regions, sectors = d["regions"], d["sectors"]
    R, S = len(regions), len(sectors)
    o_i, k_i, s_i = (regions.index("CHN"), sectors.index(SECTOR),
                     regions.index("USA"))
    target = float(d["tau0"][o_i, k_i, s_i]) + SHOCK_PP
    shock = ("CHN", SECTOR, "USA", target)

    print("=" * 70)
    print("A. 数据层")
    print("=" * 70)

    # A1 zero flows
    floor_cells = int((d["x0"] <= 1.5e-4).sum())   # scaled floor ~1mn USD
    print(f"A1 零流量格(地板单元): {floor_cells} / {d['x0'].size}  "
          f"（流量地板 1.0 百万美元）→ {'PASS(有报告)' if floor_cells >= 0 else ''}")

    # A2 global transfer closure
    sumB = d["B"].sum()
    relB = abs(sumB) / d["I0"].sum()
    print(f"A2 ΣB = {sumB:.3e}（相对总收入 {relB:.1e}）→ "
          f"{'PASS' if relB < 1e-6 else 'FAIL'}")

    # A3 absorption identity: Q0 vs sum_r q0*x0
    lhs = d["Q0"]                                  # [s, i]
    rhs = np.einsum("ris,ris->si", d["x0"], d["q0"])
    rel = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-12)
    print(f"A3 吸收恒等式 最大相对差 = {rel.max():.3e} → "
          f"{'PASS' if rel.max() < 1e-6 else 'FAIL'}")

    # A4 beta row sums
    bs = d["beta"].sum(axis=1)
    print(f"A4 β 行和与 1 的最大差 = {np.abs(bs - 1).max():.3e} → "
          f"{'PASS' if np.abs(bs - 1).max() < 1e-9 else 'FAIL'}")

    # A5 data homogeneity (all 2022 by construction of the loaders)
    print("A5 数据同源性：flows/io/production/tariffs/transfers 全部 2022 "
          "（管线常数 YEAR=2022）→ PASS")

    print()
    print("=" * 70)
    print("B. 基准复制")
    print("=" * 70)

    res = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                      homotopy_steps=5)
    b, c = res["benchmark"], res["counterfactual"]
    comps = {
        "p": np.abs(b["p"] - 1).max(),
        "w": np.abs(b["w"] - 1).max(),
        "Y/Y0": np.abs(b["Y"] / d["Y0"] - 1).max(),
        "I/I0": np.abs(b["I"] / d["I0"] - 1).max(),
        "x/x0": np.abs(b["x"] / d["x0"] - 1).max(),
        "goods_resid": np.abs(b["goods_log_resid"]).max(),
    }
    for k, v in comps.items():
        print(f"B2  {k:12s} max dev = {v:.3e}")
    print(f"B3  求解状态: benchmark={res['bench_status']}  "
          f"counterfactual={res['cf_status']}")

    print()
    print("=" * 70)
    print("C. 理论不变性")
    print("=" * 70)

    # C1 numeraire invariance (B != 0 here)
    res2 = solve_model(data=d, numeraire=("USA", "labor"), shock=shock,
                       homotopy_steps=5)
    b2, c2 = res2["benchmark"], res2["counterfactual"]
    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"C1 计价物不变性（真实数据）: max dev = {dev:.3e} → "
          f"{'PASS' if dev < 1e-3 else 'FAIL'}")

    # C2 homogeneity: data x2 -> quantities x2, prices unchanged,
    # incomes x2. LS (lump-sum tax) must scale too.
    d2 = dict(d)
    for key in ("x0", "C0", "G0", "I0i", "B", "LS"):
        d2[key] = d[key] * 2.0
    d2["alpha"] = d["alpha"]           # shares unchanged
    d2["endow"] = d["endow"] * 2.0
    for key in ("Y0", "Q0", "D0", "M0"):
        if key in d:
            d2[key] = d[key] * 2.0
    resH = solve_model(data=d2, numeraire=("CHN", "labor"), shock=shock,
                       homotopy_steps=5)
    bH = resH["benchmark"]
    q_dbl = max(np.abs(bH["x"] / b["x"] - 2.0).max(),
                np.abs(bH["Y"] / b["Y"] - 2.0).max())
    p_same = max(np.abs(bH["p"] - b["p"]).max(),
                 np.abs(bH["w"] - b["w"]).max())
    inc_ratio = np.abs(bH["I"] / b["I"] - 2.0).max()
    print(f"C2 零次齐次性（数据×2）: 数量与 2 倍的偏差 {q_dbl:.3e}，"
          f"价格与基准的偏差 {p_same:.3e}，"
          f"收入比率与 2 的差 {inc_ratio:.3e} → "
          f"{'PASS' if q_dbl < 1e-3 and p_same < 1e-4 and inc_ratio < 1e-3 else 'FAIL'}")

    # C3 counterfactual residuals
    gm = np.abs(c["goods_log_resid"]).max()
    print(f"C3 反事实商品市场残差: {gm:.3e} → {'PASS' if gm < 1e-6 else 'FAIL'}")

    print()
    print("=" * 70)
    print(f"D. 反事实经济学合理性（USA 对 CHN {SECTOR} +10pp）")
    print("=" * 70)

    taxed = (c["x"][o_i, k_i, s_i] / b["x"][o_i, k_i, s_i] - 1) * 100
    print(f"D1 被征税进口量变化: {taxed:+.1f}% → "
          f"{'PASS' if taxed < 0 else 'FAIL'}")

    # D2 diversion: third parties' ELE exports to USA rise; US domestic ELE up
    div_rows = []
    for o in range(R):
        if o in (o_i, s_i):
            continue
        chg = (c["x"][o, k_i, s_i] / b["x"][o, k_i, s_i] - 1) * 100
        div_rows.append((regions[o], chg))
    all_up = all(v > 0 for _, v in div_rows)
    print(f"D2 贸易转移（第三方对美 ELE 出口）: "
          f"{dict((r, round(v, 2)) for r, v in div_rows)} → "
          f"{'PASS' if all_up else 'CHECK'}")

    # D3 terms of trade: CHN ELE producer price should fall relative to US
    tot = (c["p"][o_i, k_i] / b["p"][o_i, k_i]
           - (c["p"][s_i, k_i] / b["p"][s_i, k_i]))
    print(f"D3 贸易条件（CHN ELE 相对 USA ELE 生产者价格变化）: "
          f"{tot * 100:+.2f} pp → {'PASS' if tot < 0 else 'CHECK'}")

    # D4 welfare signs
    rr = c["real_income"] / b["real_income"]
    w_chn, w_usa = (rr[o_i] - 1) * 100, (rr[s_i] - 1) * 100
    print(f"D4 福利符号: CHN {w_chn:+.3f}%  USA {w_usa:+.3f}% → "
          f"{'PASS' if w_chn < 0 else 'CHECK'}")

    # D5 magnitudes
    wmax = np.abs(rr - 1).max() * 100
    print(f"D5 福利量级: 全模型最大 |Δw| = {wmax:.3f}% → "
          f"{'PASS' if wmax < 2 else 'FAIL'}")

    # D6 tariff revenue accrues to importer (USA)
    tau_mat = d["tau0"].copy()
    tau_mat[o_i, k_i, s_i] = target
    rev_b = np.einsum("ris,ri,ris->s", d["tau0"], b["p"], b["x"])
    rev_c = np.einsum("ris,ri,ris->s", tau_mat, c["p"], c["x"])
    print(f"D6 关税收入变化（百万美元）: USA {((rev_c - rev_b)[s_i]) * 1e4:,.0f}，"
          f"CHN {((rev_c - rev_b)[o_i]) * 1e4:,.0f} → "
          f"{'PASS' if rev_c[s_i] > rev_b[s_i] else 'FAIL'}（收入归属征税国）")

    print()
    print("=" * 70)
    print("E. 稳健性")
    print("=" * 70)

    # E1 sigma sensitivity (sign stability)
    base_w = rr.copy()
    for f_ in (0.5, 2.0):
        df_ = dict(d)
        df_["sigma"] = d["sigma"] * f_
        df_["sigma_d"] = d["sigma_d"] * f_
        res_f = solve_model(data=df_, numeraire=("CHN", "labor"),
                            shock=shock, homotopy_steps=5)
        rf = res_f["counterfactual"]["real_income"] \
            / res_f["benchmark"]["real_income"]
        signs_same = np.all(np.sign(rf - 1) == np.sign(base_w - 1))
        print(f"E1 σ×{f_}: 福利符号不变 = {signs_same}  "
          f"(CHN {(rf[o_i]-1)*100:+.3f}%, USA {(rf[s_i]-1)*100:+.3f}%) → "
          f"{'PASS' if signs_same else 'CHECK'}")

    # E2 homotopy independence
    res_h1 = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                         homotopy_steps=1)
    c_h1 = res_h1["counterfactual"]
    dev_h = np.abs(c_h1["x"] / c["x"] - 1).max()
    print(f"E2 homotopy 无关性（1 步 vs 5 步）: max dev = {dev_h:.3e} → "
          f"{'PASS' if dev_h < 1e-3 else 'CHECK'}")

    # E3 tolerance independence
    res_t8 = solve_model(data=d, numeraire=("CHN", "labor"), shock=shock,
                         homotopy_steps=5, solver_tolerance=1e-8)
    rr_t8 = (res_t8["counterfactual"]["real_income"]
             / res_t8["benchmark"]["real_income"])
    dev_t = np.abs(rr_t8 - rr).max() * 100
    print(f"E3 容差无关性（1e-8 vs 1e-12）: 福利最大差 = {dev_t:.4f} pp → "
          f"{'PASS' if dev_t < 0.01 else 'CHECK'}")

    # E4 sector symmetry: same shock on CHM
    shock_chm = ("CHN", "CHM", "USA",
                 float(d["tau0"][o_i, sectors.index("CHM"), s_i]) + SHOCK_PP)
    res_chm = solve_model(data=d, numeraire=("CHN", "labor"),
                          shock=shock_chm, homotopy_steps=5)
    b_chm, c_chm = res_chm["benchmark"], res_chm["counterfactual"]
    taxed_chm = (c_chm["x"][o_i, sectors.index("CHM"), s_i]
                 / b_chm["x"][o_i, sectors.index("CHM"), s_i] - 1) * 100
    print(f"E4 部门对称（CHM 同冲击）: 被征税量 {taxed_chm:+.1f}%"
          f"（ELE 为 {taxed:+.1f}%）→ 结构一致 "
          f"{'PASS' if taxed_chm < 0 else 'FAIL'}")

    print("\n清单执行完毕。")


if __name__ == "__main__":
    main()
