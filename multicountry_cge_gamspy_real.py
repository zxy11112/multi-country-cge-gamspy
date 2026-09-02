# -*- coding: utf-8 -*-
"""
Real-data-only version of ``multicountry_cge_gamspy.py``.

The artificial A/B/ROW benchmark data and the 2r/3r debugging cases are
removed. Calibration always uses the real 2022 datasets:

  * dataset "io"    : CHN / USA / ROW x {prim, manu, serv}, with
                      intermediate inputs and baseline tariffs
                      (data_real_io/*.csv, built by prepare_real_data_io.py)
  * dataset "basic" : CHN / USA / ROW x {prim, manu}, no intermediates
                      (data_real/*.csv, built by prepare_real_data.py)

The model itself (MCP / PATH, Mathiesen structure) is unchanged:
  zero profit      unit cost >= p[r,i]              perp  Y[r,i]
  goods market     Y[r,i] >= sum_s x[r,i,s]         perp  p[r,i]
  factor market    endowment >= factor demand       perp  w[r,f]
  income balance   I[s] =E= factor income + tariff revenue + B[s]   perp I[s]
  definitions      P, C, Q, x, pv (auxiliary equalities)

Default experiment: USA raises its tariff on manufacturing imports from
CHN by +25 percentage points over the baseline rate.

Usage:
  python multicountry_cge_gamspy_real.py [io|basic] [delta]
  e.g.  python multicountry_cge_gamspy_real.py io 0.25
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.math import exp, log

HERE = Path(__file__).resolve().parent
# GAMS needs an ASCII-only scratch directory AND an ASCII-only process
# working directory (the project path contains non-ASCII characters,
# which breaks the GAMS runtime on Windows with
# "GetCurrentDir failed ... not ANSI").
GAMS_WORK = Path(r"D:\gamspy_demos\gams_work")
GAMS_WORK.mkdir(parents=True, exist_ok=True)
os.chdir(GAMS_WORK)


# ----------------------------------------------------------------------
# Calibration (shared by both real datasets)
# ----------------------------------------------------------------------
def calibrate(regions, sectors, factors, x0_arr, fp, sigma, B,
              io0=None, C0_in=None, tau0=None) -> dict:
    """Calibrated-share calibration; benchmark prices normalized to one.

    x0_arr : [r, i, s] benchmark bilateral quantities
    fp     : [r, i, f] benchmark factor payments
    sigma  : [i]       Armington elasticities
    B      : [r]       fixed net foreign transfers (M - X)
    io0    : [j, r, i] composite intermediate input j per unit output i
             (None -> no intermediates)
    C0_in  : [s, i]    benchmark household final demand
             (None -> households absorb everything)
    tau0   : [r, i, s] benchmark ad valorem tariffs (None -> all zero)
    """
    R, S, F = len(regions), len(sectors), len(factors)
    if tau0 is None:
        tau0 = np.zeros((R, S, R))
    Y0 = x0_arr.sum(axis=2)                      # [r, i]
    if io0 is None:
        io0 = np.zeros((S, R, S))
    ava = 1.0 - io0.sum(axis=0)                  # [r, i] value-added share
    va_val = ava * Y0                            # [r, i] value added
    afa = fp / va_val[:, :, None]                # [r, i, f] factor share in VA
    alpha = fp / Y0[:, :, None]                  # [r, i, f] (legacy key)
    endow = fp.sum(axis=1)                       # [r, f]
    q0 = 1.0 + tau0                              # [r, i, s]
    exp0 = np.einsum("ris,ris->si", q0, x0_arr)  # [s, i] destination absorption
    if C0_in is None:
        C0 = exp0.copy()
        I0 = endow.sum(axis=1) + B               # [r]
    else:
        C0 = np.asarray(C0_in, dtype=float)
        I0 = C0.sum(axis=1)                      # [r]
    beta = C0 / I0[:, None]                      # [s, i]
    # total composite demand = final + intermediate
    Q0 = C0 + np.einsum("isj,sj->si", io0, Y0)   # [s, i]
    sh0 = x0_arr * q0 / Q0.T[None, :, :]         # [r, i, s] Armington shares

    return dict(
        regions=regions, sectors=sectors, factors=factors,
        x0=x0_arr, alpha=alpha, endow=endow, q0=q0, exp0=exp0,
        I0=I0, C0=C0, beta=beta, sh0=sh0, sigma=sigma, B=B, tau0=tau0,
        Y0=Y0, io0=io0, ava=ava, afa=afa, Q0=Q0,
    )


# ----------------------------------------------------------------------
# Real datasets
# ----------------------------------------------------------------------
def get_real_data() -> dict:
    """CHN / USA / ROW x {prim, manu}, 2022, from data_real/*.csv.

    Built by prepare_real_data.py from OECD IOTs (VA) and UN Comtrade.
    Armington elasticities are illustrative literature values.
    Values are in million USD (no rescaling).
    """
    ddir = HERE / "data_real"
    regions = ["CHN", "USA", "ROW"]
    sectors = ["prim", "manu"]
    factors = ["labor", "capital"]

    prod = pd.read_csv(ddir / "production_va_2022.csv")
    flows = pd.read_csv(ddir / "bilateral_flows_2022.csv")
    transfers = pd.read_csv(ddir / "net_transfers_2022.csv")

    x0_arr = np.zeros((3, 2, 3))
    for row_ in flows.itertuples():
        x0_arr[regions.index(row_.origin), sectors.index(row_.sector),
               regions.index(row_.destination)] = row_.value

    fp = np.zeros((3, 2, 2))
    for row_ in prod.itertuples():
        fp[regions.index(row_.region), sectors.index(row_.sector), 0] = row_.labor
        fp[regions.index(row_.region), sectors.index(row_.sector), 1] = row_.capital

    sigma = np.array([3.0, 4.0])   # illustrative: prim, manu
    B = transfers.set_index("region")["net_foreign_transfer"].reindex(regions).to_numpy()

    # Y0 must equal total factor payments AND total flows per variety
    d = calibrate(regions, sectors, factors, x0_arr, fp, sigma, B)
    assert np.allclose(fp.sum(axis=2), d["Y0"], rtol=1e-9), \
        "factor payments do not add up to output"
    d["unit_scale"] = 1.0   # values are already in million USD
    return d


def get_real_data_io() -> dict:
    """CHN / USA / ROW x {prim, manu, serv}, 2022, WITH intermediates.

    Built by prepare_real_data_io.py from OECD IOTs (DOMIMP + VA) and
    UN Comtrade origin shares. Armington elasticities are illustrative.
    """
    ddir = HERE / "data_real_io"
    regions = ["CHN", "USA", "ROW"]
    sectors = ["prim", "manu", "serv"]
    factors = ["labor", "capital"]

    prod = pd.read_csv(ddir / "production_2022.csv")
    flows = pd.read_csv(ddir / "bilateral_flows_2022.csv")
    io_df = pd.read_csv(ddir / "io_coefficients_2022.csv")
    c0_df = pd.read_csv(ddir / "final_demand_2022.csv")
    transfers = pd.read_csv(ddir / "net_transfers_2022.csv")

    R, S = len(regions), len(sectors)
    x0_arr = np.zeros((R, S, R))
    for row_ in flows.itertuples():
        x0_arr[regions.index(row_.origin), sectors.index(row_.sector),
               regions.index(row_.destination)] = row_.value

    fp = np.zeros((R, S, 2))
    for row_ in prod.itertuples():
        fp[regions.index(row_.region), sectors.index(row_.sector), 0] = row_.labor
        fp[regions.index(row_.region), sectors.index(row_.sector), 1] = row_.capital

    io0 = np.zeros((S, R, S))          # [input j, region, output i]
    for row_ in io_df.itertuples():
        io0[sectors.index(row_.input), regions.index(row_.region),
            sectors.index(row_.output)] = row_.value

    C0 = np.zeros((R, S))
    for row_ in c0_df.itertuples():
        C0[regions.index(row_.region), sectors.index(row_.sector)] = row_.C0

    # elasticities from a separate input file (edit this file, not code,
    # for sensitivity analysis)
    elas_csv = ddir / "elasticities.csv"
    if elas_csv.exists():
        sigma = (pd.read_csv(elas_csv).set_index("sector")["sigma"]
                 .reindex(sectors).to_numpy())
    else:
        sigma = np.array([3.0, 4.0, 2.0])  # illustrative: prim, manu, serv

    # baseline (benchmark-year) tariffs, zero if the file is absent
    tau0 = np.zeros((R, S, R))
    tau_csv = ddir / "baseline_tariffs_2022.csv"
    if tau_csv.exists():
        for row_ in pd.read_csv(tau_csv).itertuples():
            tau0[regions.index(row_.origin), sectors.index(row_.sector),
                 regions.index(row_.destination)] = row_.rate

    B = transfers.set_index("region")["net_foreign_transfer"] \
        .reindex(regions).to_numpy()

    # Rescale to keep PATH well conditioned (raw values are ~1e7).
    # The model is homogeneous in the data units, so ratios/percentages
    # are unaffected; multiply levels by 1/SCALE to recover million USD.
    SCALE = 1e-4
    x0_arr *= SCALE
    fp *= SCALE
    C0 *= SCALE
    B = B * SCALE

    d = calibrate(regions, sectors, factors, x0_arr, fp, sigma, B,
                  io0=io0, C0_in=C0, tau0=tau0)
    # value added implied by the io matrix must equal total factor payments
    assert np.allclose(d["ava"] * d["Y0"], fp.sum(axis=2), rtol=1e-6), \
        "io-implied value added != factor payments"
    # closure: household income = factor income + tariff revenue + transfers
    T0 = (tau0 * x0_arr).sum(axis=(0, 1))
    assert np.allclose(d["I0"], d["endow"].sum(axis=1) + T0 + B, rtol=1e-6), \
        "closure identity I0 = VA + T0 + B violated"
    d["unit_scale"] = SCALE
    return d


def _df3(values: np.ndarray, regions, sectors, c1="r", c2="i", c3="s"):
    """[r, i, s] array -> long DataFrame."""
    rows = [
        (regions[a], sectors[b], regions[c], float(values[a, b, c]))
        for a in range(len(regions))
        for b in range(len(sectors))
        for c in range(len(regions))
    ]
    return pd.DataFrame(rows, columns=[c1, c2, c3, "value"])


# ----------------------------------------------------------------------
# Model construction and solve
# ----------------------------------------------------------------------
def solve_model(dataset: str = "io",
                numeraire: tuple | None = None,
                data: dict | None = None,
                shock: tuple | None = None,
                transfer_unit: tuple | None = None,
                homotopy_steps: int = 1) -> dict:
    """Build and solve the MCP model on a real dataset.

    dataset : "io" (default, with intermediates) or "basic";
              ignored when `data` is given.
    data    : pre-calibrated data dict from get_real_data*().
    shock   : (origin, sector, destination, target_rate); the tariff is
              set to this ABSOLUTE rate (the baseline rate is included in
              tau0). Defaults to USA's tariff on CHN manufacturing raised
              by +25 percentage points over the baseline.
    numeraire : (region, factor) fixed at price 1; defaults to the first
              region's labor.
    transfer_unit : (region, factor) whose price denominates the fixed net
              foreign transfers B. Must NOT depend on the numeraire choice;
              defaults to the first region's first factor. Denominating B in a
              fixed price (rather than implicitly in the numeraire) keeps the
              system homogeneous of degree zero in prices, which is required
              for numeraire invariance when B != 0.
    homotopy_steps : if > 1, approach the target tariff gradually
              (baseline -> target in equal steps, each solve warm-started
              from the previous solution). Use this for large shocks.
    """
    if data is None:
        data = get_real_data_io() if dataset == "io" else get_real_data()
    d = data
    regions, sectors, factors = d["regions"], d["sectors"], d["factors"]
    if numeraire is None:
        numeraire = (regions[0], factors[0])
    if transfer_unit is None:
        transfer_unit = (regions[0], factors[0])
    if shock is None:
        o_idx = regions.index("CHN")
        i_idx = sectors.index("manu")
        s_idx = regions.index("USA")
        target = float(d["tau0"][o_idx, i_idx, s_idx]) + 0.25
        shock = ("CHN", "manu", "USA", target)

    m = Container(working_directory=str(GAMS_WORK))
    r = Set(m, "r", records=regions)
    i = Set(m, "i", records=sectors)
    f = Set(m, "f", records=factors)
    s = Alias(m, "s", r)
    j = Alias(m, "j", i)               # input sector (intermediates)

    # ---- parameters ----
    x0 = Parameter(m, "x0", [r, i, s], records=_df3(d["x0"], regions, sectors))
    q0 = Parameter(m, "q0", [r, i, s], records=_df3(d["q0"], regions, sectors))
    sh0 = Parameter(m, "sh0", [r, i, s], records=_df3(d["sh0"], regions, sectors))
    io0 = Parameter(
        m, "io0", [j, r, i],
        records=pd.DataFrame(
            [(sectors[a], regions[b], sectors[c], float(d["io0"][a, b, c]))
             for a in range(len(sectors))
             for b in range(len(regions))
             for c in range(len(sectors))],
            columns=["j", "r", "i", "value"],
        ),
    )
    ava = Parameter(
        m, "ava", [r, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["ava"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["r", "i", "value"],
        ),
    )
    afa = Parameter(
        m, "afa", [r, i, f],
        records=pd.DataFrame(
            [(regions[a], sectors[b], factors[c], float(d["afa"][a, b, c]))
             for a in range(len(regions))
             for b in range(len(sectors))
             for c in range(len(factors))],
            columns=["r", "i", "f", "value"],
        ),
    )
    Qbar = Parameter(
        m, "Qbar", [s, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["Q0"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["s", "i", "value"],
        ),
    )
    alpha = Parameter(
        m, "alpha", [r, i, f],
        records=pd.DataFrame(
            [(regions[a], sectors[b], factors[c], float(d["alpha"][a, b, c]))
             for a in range(len(regions))
             for b in range(len(sectors))
             for c in range(len(factors))],
            columns=["r", "i", "f", "value"],
        ),
    )
    endow = Parameter(
        m, "endow", [r, f],
        records=pd.DataFrame(
            [(regions[a], factors[b], float(d["endow"][a, b]))
             for a in range(len(regions)) for b in range(len(factors))],
            columns=["r", "f", "value"],
        ),
    )
    beta = Parameter(
        m, "beta", [s, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["beta"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["s", "i", "value"],
        ),
    )
    C0 = Parameter(
        m, "C0", [s, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["C0"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["s", "i", "value"],
        ),
    )
    sigma = Parameter(m, "sigma", [i],
                      records=pd.DataFrame({"i": sectors, "value": d["sigma"]}))
    Bpar = Parameter(m, "Bpar", [s],
                     records=pd.DataFrame({"s": regions, "value": d["B"]}))
    tau = Parameter(m, "tau", [r, i, s],
                    records=_df3(d["tau0"], regions, sectors))

    # ---- variables ----
    p = Variable(m, "p", "positive", [r, i])     # producer price
    w = Variable(m, "w", "positive", [r, f])     # factor price
    Y = Variable(m, "Y", "positive", [r, i])     # output
    Pc = Variable(m, "Pc", "free", [s, i])       # Armington composite price
    C = Variable(m, "C", "free", [s, i])         # household final demand
    Q = Variable(m, "Q", "free", [s, i])         # total composite demand
    x = Variable(m, "x", "free", [r, i, s])      # bilateral quantity
    pv = Variable(m, "pv", "free", [r, i])       # value-added price index
    INC = Variable(m, "INC", "free", [s])        # household income

    # ---- initial values: the benchmark itself ----
    p.l[r, i] = 1
    w.l[r, f] = 1
    pv.l[r, i] = 1
    Y.l[r, i] = Parameter(m, "Y0par", [r, i], records=pd.DataFrame(
        [(regions[a], sectors[b], float(d["Y0"][a, b]))
         for a in range(len(regions)) for b in range(len(sectors))],
        columns=["r", "i", "value"]))[r, i]
    Pc.l[s, i] = 1
    C.l[s, i] = C0[s, i]
    Q.l[s, i] = Qbar[s, i]
    x.l[r, i, s] = x0[r, i, s]
    INC.l[s] = Parameter(m, "I0par", [s],
                       records=pd.DataFrame({"s": regions, "value": d["I0"]}))[s]

    # numeraire
    w.fx[numeraire[0], numeraire[1]] = 1

    # ---- equations ----
    profit = Equation(m, "profit", "geq", [r, i])
    goods = Equation(m, "goods", "geq", [r, i])
    factormarket = Equation(m, "factormarket", "geq", [r, f])
    income = Equation(m, "income", "eq", [s])
    Pdef = Equation(m, "Pdef", "eq", [s, i])
    Cdef = Equation(m, "Cdef", "eq", [s, i])
    Qdef = Equation(m, "Qdef", "eq", [s, i])
    pvdef = Equation(m, "pvdef", "eq", [r, i])
    xdef = Equation(m, "xdef", "eq", [r, i, s])

    # zero profit: intermediate cost + value-added cost >= producer price
    profit[r, i] = (
        Sum(j, io0[j, r, i] * Pc[r, j]) + ava[r, i] * pv[r, i]
    ) >= p[r, i]

    # value-added price index (Cobb-Douglas over factors, benchmark = 1)
    pvdef[r, i] = pv[r, i] == exp(Sum(f, afa[r, i, f] * log(w[r, f])))

    # goods market clearing
    goods[r, i] = Y[r, i] >= Sum(s, x[r, i, s])

    # factor market clearing (Shephard on the CD value-added bundle)
    factormarket[r, f] = endow[r, f] >= Sum(
        i, ava[r, i] * afa[r, i, f] * pv[r, i] * Y[r, i] / w[r, f]
    )

    # household income = factor income + tariff revenue + net transfer
    # NOTE: B is denominated in a FIXED reference price (transfer_unit),
    # independent of the numeraire; otherwise B != 0 breaks homogeneity
    # and hence numeraire invariance.
    transfer_price = w[transfer_unit[0], transfer_unit[1]]
    income[s] = INC[s] == (
        Sum(f, w[s, f] * endow[s, f])
        + Sum((r, i), tau[r, i, s] * p[r, i] * x[r, i, s])
        + Bpar[s] * transfer_price
    )

    # Armington composite price (calibrated share form, P0 = 1)
    Pdef[s, i] = Pc[s, i] == (
        Sum(r, sh0[r, i, s]
            * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s]) ** (1 - sigma[i]))
    ) ** (1 / (1 - sigma[i]))

    # Cobb-Douglas household demand
    Cdef[s, i] = C[s, i] == beta[s, i] * INC[s] / Pc[s, i]

    # total composite demand = household + Leontief intermediate demand
    Qdef[s, i] = Q[s, i] == C[s, i] + Sum(j, io0[i, s, j] * Y[s, j])

    # bilateral Armington demand (share of total composite demand)
    xdef[r, i, s] = x[r, i, s] == (
        x0[r, i, s]
        * (Q[s, i] / Qbar[s, i])
        * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s]) ** (-sigma[i])
        * Pc[s, i] ** sigma[i]
    )

    cge = Model(
        m, "cge",
        equations=[profit, goods, factormarket, income, Pdef, Cdef, Qdef,
                   pvdef, xdef],
        problem="MCP",
        matches={
            profit: Y,
            goods: p,
            factormarket: w,
            income: INC,
            Pdef: Pc,
            Cdef: C,
            Qdef: Q,
            pvdef: pv,
            xdef: x,
        },
    )

    # Tight convergence tolerance: with real data (~1e6 scale) PATH's
    # default leaves ~1e-2 relative noise, which breaks the
    # numeraire-invariance check.
    path_opts = {"convergence_tolerance": 1e-10}

    # ---------- benchmark solve (tau = tau0) ----------
    cge.solve(solver="PATH", solver_options=path_opts)
    bench = _extract(d, p, w, Y, Pc, C, x, INC)

    # ---------- counterfactual: tariff shock ----------
    # Homotopy: step the tariff gradually from its benchmark value to the
    # target; GAMSPy keeps variable levels, so each solve warm-starts
    # from the previous equilibrium.
    o_idx = regions.index(shock[0])
    i_idx = sectors.index(shock[1])
    s_idx = regions.index(shock[2])
    tau_start = float(d["tau0"][o_idx, i_idx, s_idx])
    for step in range(1, homotopy_steps + 1):
        rate = tau_start + (shock[3] - tau_start) * step / homotopy_steps
        tau[shock[0], shock[1], shock[2]] = rate
        cge.solve(solver="PATH", solver_options=path_opts)
    shock_res = _extract(d, p, w, Y, Pc, C, x, INC)

    return dict(data=d, benchmark=bench, counterfactual=shock_res)


def _extract(d, p, w, Y, Pc, C, x, INC) -> dict:
    regions, sectors, factors = d["regions"], d["sectors"], d["factors"]

    def arr3(var):
        rec = var.records.set_index(["r", "i", "s"])["level"]
        return np.array([[[rec[(a, b, c)] for c in regions]
                          for b in sectors] for a in regions])

    def arr2(var, c1, c2, l1, l2):
        rec = var.records.set_index([c1, c2])["level"]
        return np.array([[rec[(a, b)] for b in l2] for a in l1])

    pA = arr2(p, "r", "i", regions, sectors)
    wA = arr2(w, "r", "f", regions, factors)
    YA = arr2(Y, "r", "i", regions, sectors)
    PA = arr2(Pc, "s", "i", regions, sectors)
    CA = arr2(C, "s", "i", regions, sectors)
    xA = arr3(x)
    IA = INC.records.set_index("s")["level"].reindex(regions).to_numpy()

    # post-solve diagnostics
    col = np.prod(PA ** d["beta"], axis=1)       # cost of living
    real = IA / col
    gm = np.log(YA / xA.sum(axis=2))             # goods-market log residual
    return dict(p=pA, w=wA, Y=YA, P=PA, C=CA, x=xA, I=IA,
                cost_of_living=col, real_income=real, goods_log_resid=gm)


# ----------------------------------------------------------------------
# Verification (same standards as the artificial-data test suite)
# ----------------------------------------------------------------------
def verify(d, res, res2) -> None:
    """Three checks: benchmark replication, market clearing under the
    shock, and numeraire invariance (res vs res2)."""
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
    assert rep < 1e-6, "benchmark replication failed"

    gm = np.abs(c["goods_log_resid"]).max()
    print(f"[2] goods-market clearing under shock: max |log resid| = {gm:.3e}")
    assert gm < 1e-6, "goods markets do not clear"

    dev = max(
        np.abs(c["x"] / c2["x"] - 1).max(),
        np.abs(c["Y"] / c2["Y"] - 1).max(),
        np.abs(c["real_income"] / b["real_income"]
               - c2["real_income"] / b2["real_income"]).max(),
    )
    print(f"[3] numeraire invariance (real data): max dev = {dev:.3e}")
    assert dev < 1e-6, "numeraire invariance failed"
    print("All verification checks passed.\n")


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(d, res, shock, numeraire) -> None:
    b, c = res["benchmark"], res["counterfactual"]
    regions, sectors = d["regions"], d["sectors"]
    R, S = len(regions), len(sectors)
    us = 1.0 / d["unit_scale"]      # levels -> million USD

    o_idx, i_idx, s_idx = (regions.index(shock[0]), sectors.index(shock[1]),
                           regions.index(shock[2]))
    base_rate = float(d["tau0"][o_idx, i_idx, s_idx])
    print(f"=== shock: {shock[2]} tariff on {shock[1]} from {shock[0]}: "
          f"{base_rate:.2%} -> {shock[3]:.2%} ===\n")

    # ---------- bilateral flow changes (all sectors) ----------
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

    # destination's absorption of the shocked sector, by origin
    div = pd.DataFrame({
        "origin": regions,
        "benchmark_mUSD": b["x"][:, i_idx, s_idx] * us,
        "counterfactual_mUSD": c["x"][:, i_idx, s_idx] * us,
        "pct_change": (c["x"][:, i_idx, s_idx] / b["x"][:, i_idx, s_idx] - 1) * 100,
    })
    print(f"\n{shock[2]} {shock[1]} absorption by origin:")
    print(div.round(2).to_string(index=False))

    # ---------- welfare ----------
    rr = c["real_income"] / b["real_income"]
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * us * (rr - 1),
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
         for o in range(R) for k in range(len(d["factors"]))],
        columns=["region", "factor", "price_pct"])
    print(f"\nfactor price changes (numeraire: {numeraire[0]} {numeraire[1]}):")
    print(fac.round(3).to_string(index=False))


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "io"
    delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    d = get_real_data_io() if dataset == "io" else get_real_data()
    regions, sectors = d["regions"], d["sectors"]

    # shock: baseline tariff + delta on CHN -> USA manufacturing
    o_i, k_i, s_i = (regions.index("CHN"), sectors.index("manu"),
                     regions.index("USA"))
    target = float(d["tau0"][o_i, k_i, s_i]) + delta
    shock = ("CHN", "manu", "USA", target)
    print(f"dataset: {dataset}; shock: USA raises tariff on manu from CHN "
          f"from {d['tau0'][o_i, k_i, s_i]:.2%} to {target:.2%} "
          f"(+{delta:.0%} pp)\n")

    num1, num2 = ("CHN", "labor"), ("USA", "labor")
    res = solve_model(data=d, numeraire=num1, shock=shock, homotopy_steps=5)
    res2 = solve_model(data=d, numeraire=num2, shock=shock, homotopy_steps=5)
    verify(d, res, res2)
    report(d, res, shock, num1)


if __name__ == "__main__":
    main()
