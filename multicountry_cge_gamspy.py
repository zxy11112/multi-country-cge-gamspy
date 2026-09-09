# -*- coding: utf-8 -*-
"""
GAMSPy (MCP / PATH) port of ``multicountry_cge.py``.

Same model, same artificial data, same tariff experiment:
  * regions A, B, (ROW); sectors food, manufacturing; factors labor, capital
  * Cobb-Douglas production, one Cobb-Douglas household per region
  * one-level Armington aggregation across origins (calibrated share form)
  * ad valorem tariffs rebated lump-sum to the destination household
  * numeraire w[A, labor] = 1

MCP structure (Mathiesen style):
  zero profit      unit cost >= p[r,i]              perp  Y[r,i]
  goods market     Y[r,i] >= sum_s x[r,i,s]         perp  p[r,i]
  factor market    endowment >= factor demand       perp  w[r,f]
  income balance   I[s] =E= factor income + tariff revenue + B[s]   perp I[s]
  definitions      P, C, x (auxiliary equalities)   perp  P, C, x

The script solves the benchmark, checks replication, applies the 10%
tariff on A's manufacturing imports from B, and compares every key
number against the pre-generated results of the original scipy version
in ``outputs/<case>/``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Ord,
    Parameter,
    Set,
    Sum,
    Variable,
)
from gamspy.math import exp, log

HERE = Path(__file__).resolve().parent
# GAMS needs an ASCII-only scratch directory AND an ASCII-only process
# working directory (the project path may contain non-ASCII characters,
# which breaks the GAMS runtime with
# "GetCurrentDir failed ... not ANSI").
# Pick a platform-appropriate ASCII fallback; override with GAMSPY_WORKDIR.
if os.environ.get("GAMSPY_WORKDIR"):
    GAMS_WORK = Path(os.environ["GAMSPY_WORKDIR"])
elif sys.platform.startswith("win"):
    GAMS_WORK = Path(r"D:\gamspy_demos\gams_work")
else:
    GAMS_WORK = Path("/tmp/gamspy_work")
GAMS_WORK.mkdir(parents=True, exist_ok=True)
os.chdir(GAMS_WORK)


# ----------------------------------------------------------------------
# Artificial benchmark data (identical to the scipy version)
# ----------------------------------------------------------------------
def get_data(case: str) -> dict:
    sectors = ["food", "manufacturing"]
    factors = ["labor", "capital"]
    if case == "2r":
        regions = ["A", "B"]
        x0 = {
            "food": np.array([[45.0, 15.0], [15.0, 15.0]]),
            "manufacturing": np.array([[25.0, 15.0], [15.0, 55.0]]),
        }
        fp = np.array(
            [[[42.0, 18.0], [20.0, 20.0]], [[18.0, 12.0], [28.0, 42.0]]]
        )
    elif case == "3r":
        regions = ["A", "B", "ROW"]
        x0 = {
            "food": np.array(
                [[42.0, 8.0, 10.0], [8.0, 14.0, 8.0], [10.0, 8.0, 72.0]]
            ),
            "manufacturing": np.array(
                [[24.0, 8.0, 8.0], [8.0, 50.0, 12.0], [8.0, 12.0, 90.0]]
            ),
        }
        fp = np.array(
            [
                [[42.0, 18.0], [20.0, 20.0]],
                [[18.0, 12.0], [28.0, 42.0]],
                [[63.0, 27.0], [44.0, 66.0]],
            ]
        )
    else:
        raise ValueError("case must be '2r' or '3r'")

    R, S, F = len(regions), len(sectors), len(factors)
    x0_arr = np.stack([x0[sec] for sec in sectors], axis=1)  # [r, i, s]
    sigma = np.array([2.0, 4.0])
    B = np.zeros(R)
    return calibrate(regions, sectors, factors, x0_arr, fp, sigma, B)


def calibrate(regions, sectors, factors, x0_arr, fp, sigma, B,
              io0=None, C0_in=None, tau0=None, sigma_d=None,
              G0_in=None, I0_in=None) -> dict:
    """Calibrated-share calibration; benchmark prices normalized to one.

    x0_arr : [r, i, s] benchmark bilateral quantities
    fp     : [r, i, f] benchmark factor payments
    sigma  : [i]       Armington elasticities (flat: across all origins;
             nested: across FOREIGN origins, i.e. the bottom nest)
    B      : [r]       fixed net foreign transfers (M - X)
    io0    : [j, r, i] composite intermediate input j per unit output i
             (None -> no intermediates, the original prototype)
    C0_in  : [s, i]    benchmark HOUSEHOLD final demand
             (None -> households absorb everything, prototype behavior)
    tau0   : [r, i, s] benchmark ad valorem tariffs (None -> all zero)
    sigma_d: [i]       domestic-vs-import elasticity (top nest). When
             given, the demand system is nested Armington: the bottom nest
             aggregates foreign origins (sigma), the top nest composites
             the domestic variety against the import aggregate (sigma_d).
    G0_in  : [s, i]    government consumption (None -> no government)
    I0_in  : [s, i]    investment demand (None -> no savings/investment)
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
        # prototype: households absorb everything, no government/saving
        C0 = exp0.copy()
        I0 = endow.sum(axis=1) + B               # [r]
        G0 = np.zeros((R, S))
        I0i = np.zeros((R, S))
        LS = np.zeros(R)
        mps = np.zeros(R)
        gov = False
    elif G0_in is None:
        # v3 style: households absorb all final demand incl. gov/investment,
        # tariff revenue rebated to households (no explicit government)
        C0 = np.asarray(C0_in, dtype=float)
        I0 = C0.sum(axis=1)                      # [r]
        G0 = np.zeros((R, S))
        I0i = np.zeros((R, S))
        LS = np.zeros(R)
        mps = np.zeros(R)
        gov = False
    else:
        # v5: government + savings/investment
        C0 = np.asarray(C0_in, dtype=float)
        G0 = np.asarray(G0_in, dtype=float)      # government consumption
        I0i = np.asarray(I0_in, dtype=float)     # investment
        T0 = (tau0 * x0_arr).sum(axis=(0, 1))    # [s] benchmark tariff revenue
        LS = G0.sum(axis=1) - T0                 # lump-sum tax (residual)
        YH0 = endow.sum(axis=1) + B - LS         # household income
        mps = I0i.sum(axis=1) / YH0              # savings rate (savings-driven)
        # closure: C0+G0+I0 = VA + T0 + B  <=>  (1-mps)*YH0 = C0_total
        # atol: data are scaled by 1e-4, so 0.01 = 100 million USD;
        # covers the small residual from positive-flow flooring.
        assert np.allclose(C0.sum(axis=1) + G0.sum(axis=1) + I0i.sum(axis=1),
                           endow.sum(axis=1) + T0 + B, rtol=1e-6, atol=0.01), \
            "closure identity C0+G0+I0 = VA + T0 + B violated"
        I0 = YH0                                 # [r] household income
        gov = True
    beta = C0 / C0.sum(axis=1)[:, None]          # [s, i]
    gshare = np.divide(G0, G0.sum(axis=1)[:, None],
                       out=np.zeros_like(G0), where=G0.sum(axis=1)[:, None] != 0)
    ishare = np.divide(I0i, I0i.sum(axis=1)[:, None],
                       out=np.zeros_like(I0i),
                       where=I0i.sum(axis=1)[:, None] != 0)
    # total composite demand = final + intermediate
    Q0 = C0 + G0 + I0i + np.einsum("isj,sj->si", io0, Y0)   # [s, i]
    sh0 = x0_arr * q0 / Q0.T[None, :, :]         # [r, i, s] Armington shares

    out = dict(
        regions=regions, sectors=sectors, factors=factors,
        x0=x0_arr, alpha=alpha, endow=endow, q0=q0, exp0=exp0,
        I0=I0, C0=C0, beta=beta, sh0=sh0, sigma=sigma, B=B, tau0=tau0,
        Y0=Y0, io0=io0, ava=ava, afa=afa, Q0=Q0,
        G0=G0, I0i=I0i, LS=LS, mps=mps, gshare=gshare, ishare=ishare,
        gov=gov,
    )

    if sigma_d is not None:
        # nested Armington calibration
        D0 = np.zeros((R, S))
        for s_ in range(R):
            D0[s_] = x0_arr[s_, :, s_]           # domestic flow (diagonal)
        M0 = Q0 - D0                             # import aggregate (purchaser)
        sm0 = x0_arr * q0 / M0.T[None, :, :]     # [r,i,s] origin shares in imports
        for s_ in range(R):
            sm0[s_, :, s_] = 0.0                 # diagonal excluded by def
        sd0 = D0 / Q0                            # domestic share of total use
        out.update(sigma_d=np.asarray(sigma_d, dtype=float),
                   D0=D0, M0=M0, sm0=sm0, sd0=sd0)
    return out


def get_real_data() -> dict:
    """CHN / USA / ROW x {prim, manu}, 2022, from data_real/*.csv.

    Built by prepare_real_data.py from OECD IOTs (VA) and UN Comtrade.
    Armington elasticities are illustrative literature values.
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
    return d


def get_real_data_io(ddir_name: str = "data_real_io",
                     regions: list | None = None,
                     sectors: list | None = None,
                     sigma_default=(3.0, 4.0, 2.0),
                     year: int = 2022) -> dict:
    """Real-data IO model loader (benchmark year via file suffix).

    Reads <ddir_name>/{production,bilateral_flows,io_coefficients,
    final_demand,net_transfers,baseline_tariffs,elasticities}_<year>.csv
    built by the corresponding prepare_* pipeline.
    Defaults: CHN / USA / ROW x {prim, manu, serv}, year 2022.
    """
    ddir = HERE / ddir_name
    if regions is None:
        regions = ["CHN", "USA", "ROW"]
    if sectors is None:
        sectors = ["prim", "manu", "serv"]
    factors = ["labor", "capital"]

    prod = pd.read_csv(ddir / f"production_{year}.csv")
    flows = pd.read_csv(ddir / f"bilateral_flows_{year}.csv")
    io_df = pd.read_csv(ddir / f"io_coefficients_{year}.csv")
    c0_df = pd.read_csv(ddir / f"final_demand_{year}.csv")
    transfers = pd.read_csv(ddir / f"net_transfers_{year}.csv")

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
    G0 = np.zeros((R, S))
    I0i = np.zeros((R, S))
    for row_ in c0_df.itertuples():
        idx = (regions.index(row_.region), sectors.index(row_.sector))
        C0[idx] = row_.C0
        G0[idx] = row_.G0
        I0i[idx] = row_.I0

    # elasticities from a separate input file (edit this file, not code,
    # for sensitivity analysis). sigma = bottom nest (across foreign
    # origins); optional sigma_d = top nest (domestic vs import aggregate)
    # -> nested Armington when present.
    elas_csv = ddir / "elasticities.csv"
    sigma_d = None
    if elas_csv.exists():
        elas = pd.read_csv(elas_csv).set_index("sector").reindex(sectors)
        sigma = elas["sigma"].to_numpy()
        if "sigma_d" in elas.columns:
            sigma_d = elas["sigma_d"].to_numpy()
    else:
        sigma = np.asarray(sigma_default, dtype=float)

    # baseline (benchmark-year) tariffs: prefer the effective (G1151-
    # anchored) version when present; zero if no tariff file at all
    tau0 = np.zeros((R, S, R))
    tau_csv_eff = ddir / f"baseline_tariffs_effective_{year}.csv"
    tau_csv = ddir / f"baseline_tariffs_{year}.csv"
    tau_file = tau_csv_eff if tau_csv_eff.exists() else tau_csv
    if tau_file.exists():
        for row_ in pd.read_csv(tau_file).itertuples():
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
    G0 *= SCALE
    I0i *= SCALE
    B = B * SCALE

    d = calibrate(regions, sectors, factors, x0_arr, fp, sigma, B,
                  io0=io0, C0_in=C0, tau0=tau0, sigma_d=sigma_d,
                  G0_in=G0, I0_in=I0i)
    # value added implied by the io matrix must equal total factor payments
    # (atol: data are scaled by 1e-4, so 0.01 = 100 million USD)
    assert np.allclose(d["ava"] * d["Y0"], fp.sum(axis=2),
                       rtol=1e-6, atol=0.01), \
        "io-implied value added != factor payments"
    # closure is asserted inside calibrate(): C0+G0+I0 = VA + T0 + B
    return d


REGIONS_6X12 = ["CHN", "USA", "JPN", "KOR", "EU27", "ROW"]
SECTORS_6X12 = ["AGF", "MIN", "ENR", "CHM", "TXL", "WDP",
                "NMM", "MAC", "ELE", "VEH", "OTM", "SRV"]
REGIONS_14X12 = ["CHN", "USA", "JPN", "KOR", "EU27", "VNM", "THA", "MYS",
                 "IDN", "SGP", "PHL", "IND", "MEX", "ROW"]


def get_real_data_6x12() -> dict:
    """6-region x 12-sector model, 2022 (prepare_real_data_6x12.py)."""
    return get_real_data_io(
        "data_real_6x12", regions=REGIONS_6X12, sectors=SECTORS_6X12,
        sigma_default=[3.0, 4.0, 5.0, 4.0, 5.0, 4.0,
                       4.0, 5.0, 6.0, 5.0, 4.0, 2.0])


def get_real_data_14x12() -> dict:
    """14-region x 12-sector model, 2022 (prepare_real_data_14x12.py)."""
    return get_real_data_io(
        "data_real_14x12", regions=REGIONS_14X12, sectors=SECTORS_6X12,
        sigma_default=[3.0, 4.0, 5.0, 4.0, 5.0, 4.0,
                       4.0, 5.0, 6.0, 5.0, 4.0, 2.0])


REGIONS_G20 = ["CHN", "USA", "EU27", "JPN", "KOR", "IND", "CAN", "MEX",
               "BRA", "ZAF", "RUS", "SAU", "AUS", "ROW"]


def get_real_data_g20(ddir_name: str = "data_real_g20",
                      year: int = 2022) -> dict:
    """G20-flavor 14-region x 12-sector model (benchmark year set by the
    data directory and file suffix, e.g. data_real_g20_2005)."""
    return get_real_data_io(
        ddir_name, regions=REGIONS_G20, sectors=SECTORS_6X12,
        sigma_default=[3.0, 4.0, 5.0, 4.0, 5.0, 4.0,
                       4.0, 5.0, 6.0, 5.0, 4.0, 2.0],
        year=year)


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
def solve_model(case: str = "3r", tariff_rate: float = 0.10,
                numeraire=("A", "labor"), data: dict | None = None,
                shock: tuple | None = None,
                transfer_unit: tuple | None = None,
                homotopy_steps: int = 1,
                solver_tolerance: float = 1e-10) -> dict:
    """Build and solve the MCP model.

    data  : pre-calibrated data dict (defaults to the artificial `case`)
    shock : (origin, sector, destination, rate), or a list of such
            tuples for simultaneous shocks; defaults to the
            artificial case's B -> A manufacturing tariff.
    transfer_unit : (region, factor) whose price denominates the fixed net
            foreign transfers B. Must NOT depend on the numeraire choice;
            defaults to the first region's first factor. Denominating B in a
            fixed price (rather than implicitly in the numeraire) keeps the
            system homogeneous of degree zero in prices, which is required
            for numeraire invariance when B != 0.
    homotopy_steps : if > 1, approach the target tariff gradually
            (0 -> rate in equal steps, each solve warm-started from the
            previous solution). Use this for large shocks.
    """
    t_start = time.perf_counter()
    d = data if data is not None else get_data(case)
    if shock is None:
        shock = ("B", "manufacturing", "A", tariff_rate)

    h = build_model(d, numeraire=numeraire, transfer_unit=transfer_unit,
                    solver_tolerance=solver_tolerance)
    cge = h["cge"]
    tau = h["tau"]
    regions, sectors = h["regions"], h["sectors"]
    V = h["vars"]
    path_opts = {"convergence_tolerance": solver_tolerance}
    t_build_done = time.perf_counter()

    # ---------- benchmark solve (tau = 0) ----------
    t_bench_start = time.perf_counter()
    cge.solve(solver="PATH", solver_options=path_opts)
    t_bench_done = time.perf_counter()
    bench = _extract(d, **V)
    bench_status = str(cge.status)

    # ---------- counterfactual: configurable tariff shock ----------
    # Homotopy: step the tariff gradually from its benchmark value to the
    # target; GAMSPy keeps variable levels, so each solve warm-starts
    # from the previous equilibrium.
    shock_list = [shock] if isinstance(shock[0], str) else list(shock)
    parsed = []
    for shk in shock_list:
        parsed.append((regions.index(shk[0]), sectors.index(shk[1]),
                       regions.index(shk[2]), float(shk[3])))
    starts = [float(d["tau0"][o, i_, s_]) for o, i_, s_, _ in parsed]
    step_times = []
    for step in range(1, homotopy_steps + 1):
        t_step_start = time.perf_counter()
        for (o, i_, s_, target), tau_start in zip(parsed, starts):
            rate = tau_start + (target - tau_start) * step / homotopy_steps
            tau[regions[o], sectors[i_], regions[s_]] = rate
        cge.solve(solver="PATH", solver_options=path_opts)
        t_step_done = time.perf_counter()
        step_times.append(t_step_done - t_step_start)
    t_cf_done = time.perf_counter()
    shock_res = _extract(d, **V)

    t_total = time.perf_counter() - t_start
    timing = {
        "total_seconds": t_total,
        "build_seconds": h["build_seconds"],
        "benchmark_solve_seconds": t_bench_done - t_bench_start,
        "counterfactual_solve_seconds": t_cf_done - t_bench_done,
        "per_step_seconds": step_times,
        "homotopy_steps": homotopy_steps,
    }

    return dict(data=d, benchmark=bench, counterfactual=shock_res,
                bench_status=bench_status, cf_status=str(cge.status),
                timing=timing)


def build_model(d, numeraire=("A", "labor"), transfer_unit=None,
                solver_tolerance=1e-10) -> dict:
    """Build the MCP model once and return handles for repeated solves.

    The container is initialized at the benchmark data ``d`` (tau = d["tau0"]).
    Between solves, mutate the returned ``tau`` parameter (e.g. via
    ``tau.setRecords(df)``); variable levels persist across ``cge.solve()``
    calls, so each solve warm-starts from the previous equilibrium.
    """
    regions, sectors, factors = d["regions"], d["sectors"], d["factors"]
    if transfer_unit is None:
        transfer_unit = (regions[0], factors[0])
    _t_build_start = time.perf_counter()

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

    # ---- v5 parameters: government & savings/investment ----
    gov = bool(d.get("gov", False))      # False for prototype/flat cases
    mps = Parameter(m, "mps", [s],
                    records=pd.DataFrame({"s": regions, "value": d["mps"]}))
    LS = Parameter(m, "LS", [s],
                   records=pd.DataFrame({"s": regions, "value": d["LS"]}))
    gshare = Parameter(
        m, "gshare", [s, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["gshare"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["s", "i", "value"]))
    ishare = Parameter(
        m, "ishare", [s, i],
        records=pd.DataFrame(
            [(regions[a], sectors[b], float(d["ishare"][a, b]))
             for a in range(len(regions)) for b in range(len(sectors))],
            columns=["s", "i", "value"]))

    # ---- nested Armington parameters (v4) ----
    nested = d.get("sigma_d") is not None
    if nested:
        sigd = Parameter(m, "sigd", [i],
                         records=pd.DataFrame({"i": sectors,
                                               "value": d["sigma_d"]}))
        sm0 = Parameter(m, "sm0", [r, i, s],
                        records=_df3(d["sm0"], regions, sectors))
        D0par = Parameter(
            m, "D0par", [s, i],
            records=pd.DataFrame(
                [(regions[a], sectors[b], float(d["D0"][a, b]))
                 for a in range(len(regions)) for b in range(len(sectors))],
                columns=["s", "i", "value"]))
        M0par = Parameter(
            m, "M0par", [s, i],
            records=pd.DataFrame(
                [(regions[a], sectors[b], float(d["M0"][a, b]))
                 for a in range(len(regions)) for b in range(len(sectors))],
                columns=["s", "i", "value"]))
        sd0 = Parameter(
            m, "sd0", [s, i],
            records=pd.DataFrame(
                [(regions[a], sectors[b], float(d["sd0"][a, b]))
                 for a in range(len(regions)) for b in range(len(sectors))],
                columns=["s", "i", "value"]))

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
    GS = Variable(m, "GS", "free", [s])          # government spending
    SAV = Variable(m, "SAV", "free", [s])        # saving = investment
    G = Variable(m, "G", "free", [s, i])         # government demand
    INV = Variable(m, "INV", "free", [s, i])     # investment demand
    if nested:
        Dv = Variable(m, "Dv", "free", [s, i])   # domestic variety demand
        Mimp = Variable(m, "Mimp", "free", [s, i])  # import aggregate qty
        PM = Variable(m, "PM", "free", [s, i])   # import aggregate price

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
    GS.l[s] = Parameter(m, "GS0par", [s], records=pd.DataFrame(
        {"s": regions, "value": d["G0"].sum(axis=1)}))[s]
    SAV.l[s] = Parameter(m, "SAV0par", [s], records=pd.DataFrame(
        {"s": regions, "value": d["I0i"].sum(axis=1)}))[s]
    G.l[s, i] = Parameter(m, "G0par", [s, i], records=pd.DataFrame(
        [(regions[a], sectors[b], float(d["G0"][a, b]))
         for a in range(len(regions)) for b in range(len(sectors))],
        columns=["s", "i", "value"]))[s, i]
    INV.l[s, i] = Parameter(m, "I0ipar", [s, i], records=pd.DataFrame(
        [(regions[a], sectors[b], float(d["I0i"][a, b]))
         for a in range(len(regions)) for b in range(len(sectors))],
        columns=["s", "i", "value"]))[s, i]
    if nested:
        Dv.l[s, i] = D0par[s, i]
        Mimp.l[s, i] = M0par[s, i]
        PM.l[s, i] = 1
        # the diagonal of x (domestic flows) is carried by Dv in nested mode
        for reg in regions:
            x.fx[reg, i, reg] = 0

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
    GSdef = Equation(m, "GSdef", "eq", [s])
    SAVdef = Equation(m, "SAVdef", "eq", [s])
    Gdef = Equation(m, "Gdef", "eq", [s, i])
    INVdef = Equation(m, "INVdef", "eq", [s, i])

    # zero profit: intermediate cost + value-added cost >= producer price
    profit[r, i] = (
        Sum(j, io0[j, r, i] * Pc[r, j]) + ava[r, i] * pv[r, i]
    ) >= p[r, i]

    # value-added price index (Cobb-Douglas over factors, benchmark = 1)
    pvdef[r, i] = pv[r, i] == exp(Sum(f, afa[r, i, f] * log(w[r, f])))

    # ---- goods market clearing ----
    if nested:
        # domestic variety sales + exports (x diagonal is fixed to zero)
        goods[r, i] = Y[r, i] >= Dv[r, i] + Sum(s, x[r, i, s])
    else:
        goods[r, i] = Y[r, i] >= Sum(s, x[r, i, s])

    # factor market clearing (Shephard on the CD value-added bundle)
    factormarket[r, f] = endow[r, f] >= Sum(
        i, ava[r, i] * afa[r, i, f] * pv[r, i] * Y[r, i] / w[r, f]
    )

    # ---- v5: household income / government budget / savings ----
    # NOTE: B and the lump-sum tax LS are denominated in a FIXED reference
    # price (transfer_unit), independent of the numeraire; otherwise
    # nonzero transfers break homogeneity and hence numeraire invariance.
    transfer_price = w[transfer_unit[0], transfer_unit[1]]
    if gov:
        # household disposable income = factor income + transfers - lump tax
        income[s] = INC[s] == (
            Sum(f, w[s, f] * endow[s, f])
            + Bpar[s] * transfer_price
            - LS[s] * transfer_price
        )
        # government spending = tariff revenue + lump-sum tax revenue
        GSdef[s] = GS[s] == (
            Sum((r, i), tau[r, i, s] * p[r, i] * x[r, i, s])
            + LS[s] * transfer_price
        )
    else:
        # prototype/flat: tariff revenue rebated directly to households
        income[s] = INC[s] == (
            Sum(f, w[s, f] * endow[s, f])
            + Sum((r, i), tau[r, i, s] * p[r, i] * x[r, i, s])
            + Bpar[s] * transfer_price
        )
        GSdef[s] = GS[s] == LS[s] * transfer_price
    # savings-driven investment closure: fixed saving rate out of income
    SAVdef[s] = SAV[s] == mps[s] * INC[s]

    if nested:
        Ddef = Equation(m, "Ddef", "eq", [s, i])
        Mdef = Equation(m, "Mdef", "eq", [s, i])
        PMdef = Equation(m, "PMdef", "eq", [s, i])

        # top nest: composite price over domestic variety vs import aggregate
        Pdef[s, i] = Pc[s, i] == (
            sd0[s, i] * p[s, i] ** (1 - sigd[i])
            + (1 - sd0[s, i]) * PM[s, i] ** (1 - sigd[i])
        ) ** (1 / (1 - sigd[i]))

        # bottom nest: import aggregate price over foreign origins
        PMdef[s, i] = PM[s, i] == (
            Sum(r, sm0[r, i, s]
                * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s])
                ** (1 - sigma[i]))
        ) ** (1 / (1 - sigma[i]))

        # top-nest demand: domestic variety and import aggregate
        Ddef[s, i] = Dv[s, i] == (
            D0par[s, i] * (Q[s, i] / Qbar[s, i])
            * p[s, i] ** (-sigd[i]) * Pc[s, i] ** sigd[i]
        )
        Mdef[s, i] = Mimp[s, i] == (
            M0par[s, i] * (Q[s, i] / Qbar[s, i])
            * PM[s, i] ** (-sigd[i]) * Pc[s, i] ** sigd[i]
        )

        # bottom-nest bilateral import demand (off-diagonal only; the
        # diagonal of x is fixed to zero and carried by Dv)
        xdef[r, i, s].where[Ord(r) != Ord(s)] = x[r, i, s] == (
            x0[r, i, s]
            * (Mimp[s, i] / M0par[s, i])
            * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s]) ** (-sigma[i])
            * PM[s, i] ** sigma[i]
        )
    else:
        # Armington composite price (calibrated share form, P0 = 1)
        Pdef[s, i] = Pc[s, i] == (
            Sum(r, sh0[r, i, s]
                * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s]) ** (1 - sigma[i]))
        ) ** (1 / (1 - sigma[i]))

        # bilateral Armington demand (share of total composite demand)
        xdef[r, i, s] = x[r, i, s] == (
            x0[r, i, s]
            * (Q[s, i] / Qbar[s, i])
            * ((1 + tau[r, i, s]) * p[r, i] / q0[r, i, s]) ** (-sigma[i])
            * Pc[s, i] ** sigma[i]
        )

    # Cobb-Douglas household demand (out of after-saving income)
    Cdef[s, i] = C[s, i] == beta[s, i] * (1 - mps[s]) * INC[s] / Pc[s, i]

    # government demand: Cobb-Douglas over sectors out of government budget
    Gdef[s, i] = G[s, i] == gshare[s, i] * GS[s] / Pc[s, i]

    # investment demand: Cobb-Douglas over sectors out of savings
    INVdef[s, i] = INV[s, i] == ishare[s, i] * SAV[s] / Pc[s, i]

    # total composite demand = household + government + investment
    #                          + Leontief intermediate demand
    Qdef[s, i] = Q[s, i] == (
        C[s, i] + G[s, i] + INV[s, i] + Sum(j, io0[i, s, j] * Y[s, j])
    )

    eqs = [profit, goods, factormarket, income, Pdef, Cdef, Qdef, pvdef,
           xdef, GSdef, SAVdef, Gdef, INVdef]
    match_map = {
        profit: Y,
        goods: p,
        factormarket: w,
        income: INC,
        Pdef: Pc,
        Cdef: C,
        Qdef: Q,
        pvdef: pv,
        xdef: x,
        GSdef: GS,
        SAVdef: SAV,
        Gdef: G,
        INVdef: INV,
    }
    if nested:
        eqs += [Ddef, Mdef, PMdef]
        match_map.update({Ddef: Dv, Mdef: Mimp, PMdef: PM})

    cge = Model(
        m, "cge",
        equations=eqs,
        problem="MCP",
        matches=match_map,
    )
    return dict(
        m=m, cge=cge, tau=tau, regions=regions, sectors=sectors,
        factors=factors, nested=nested,
        build_seconds=time.perf_counter() - _t_build_start,
        vars=dict(p=p, w=w, Y=Y, Pc=Pc, C=C, x=x, INC=INC,
                  Dv=Dv if nested else None),
    )


def _extract(d, p, w, Y, Pc, C, x, INC, Dv=None) -> dict:
    regions, sectors, factors = d["regions"], d["sectors"], d["factors"]
    R, S, F = len(regions), len(sectors), len(factors)

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
    if Dv is not None:
        # nested mode: domestic flows live in Dv; put them on x's diagonal
        DA = arr2(Dv, "s", "i", regions, sectors)
        for k in range(R):
            xA[k, :, k] = DA[k, :]
    IA = INC.records.set_index("s")["level"].reindex(regions).to_numpy()

    # post-solve diagnostics (same formulas as the scipy version)
    col = np.prod(PA ** d["beta"], axis=1)       # cost of living
    real = IA * (1.0 - d.get("mps", 0.0)) / col  # real consumption income
    gm = np.log(YA / xA.sum(axis=2))             # goods-market log residual

    # GDP per region: nominal = factor income at current factor prices;
    # real = value added at benchmark factor prices (pv = 1).
    gdp_nom = (wA * d["endow"]).sum(axis=1)
    gdp_real = (d["ava"] * YA).sum(axis=1)

    return dict(p=pA, w=wA, Y=YA, P=PA, C=CA, x=xA, I=IA,
                cost_of_living=col, real_income=real, goods_log_resid=gm,
                gdp_nominal=gdp_nom, gdp_real=gdp_real)


# ----------------------------------------------------------------------
# Reporting and comparison with the original scipy outputs
# ----------------------------------------------------------------------
def compare(case: str, res: dict) -> None:
    d, b, c = res["data"], res["benchmark"], res["counterfactual"]
    regions, sectors = d["regions"], d["sectors"]
    ref_dir = HERE / "outputs" / ("two_region" if case == "2r" else "three_region")

    # ---- benchmark replication ----
    rep_err = max(
        np.abs(b["p"] - 1).max(),
        np.abs(b["w"] - 1).max(),
        np.abs(b["Y"] / d["Y0"] - 1).max(),
        np.abs(b["I"] / d["I0"] - 1).max(),
        np.abs(b["x"] / d["x0"] - 1).max(),
        np.abs(b["goods_log_resid"]).max(),
    )
    print(f"\n=== case {case}: benchmark replication ===")
    print(f"max deviation from benchmark data : {rep_err:.3e}")
    assert rep_err < 1e-6, "benchmark replication failed"

    # ---- A's manufacturing demand by origin (trade diversion) ----
    iA, iM = regions.index("A"), sectors.index("manufacturing")
    chg = (c["x"][:, iM, iA] / b["x"][:, iM, iA] - 1) * 100
    print(f"\n=== case {case}: A's manufacturing demand, % change by origin ===")
    out = pd.DataFrame({"origin": regions, "gamspy_pct": chg})
    if (ref_dir / "bilateral_trade_summary.csv").exists():
        ref = pd.read_csv(ref_dir / "bilateral_trade_summary.csv")
        ref = ref[(ref["destination"] == "A") & (ref["sector"] == "manufacturing")]
        out = out.merge(
            ref[["origin", "quantity_pct_change"]], on="origin", how="left"
        ).rename(columns={"quantity_pct_change": "scipy_ref_pct"})
        out["diff"] = out["gamspy_pct"] - out["scipy_ref_pct"]
    print(out.round(4).to_string(index=False))

    # ---- welfare ----
    real_ratio = c["real_income"] / b["real_income"]
    welfare = (real_ratio - 1) * 100
    ev = d["I0"] * (real_ratio - 1)
    print(f"\n=== case {case}: welfare ===")
    wdf = pd.DataFrame(
        {"region": regions, "gamspy_welfare_pct": welfare, "gamspy_EV": ev}
    )
    if (ref_dir / "country_summary.csv").exists():
        ref = pd.read_csv(ref_dir / "country_summary.csv")
        wdf = wdf.merge(
            ref[["region", "welfare_pct_change", "equivalent_variation"]],
            on="region", how="left",
        ).rename(columns={
            "welfare_pct_change": "scipy_ref_pct",
            "equivalent_variation": "scipy_ref_EV",
        })
        wdf["diff_pct"] = wdf["gamspy_welfare_pct"] - wdf["scipy_ref_pct"]
        wdf["diff_EV"] = wdf["gamspy_EV"] - wdf["scipy_ref_EV"]
    print(wdf.round(4).to_string(index=False))

    # ---- goods-market residuals after the shock ----
    print(f"\nmax |goods-market log residual| (counterfactual): "
          f"{np.abs(c['goods_log_resid']).max():.3e}")


def numeraire_invariance(case: str = "3r") -> None:
    a = solve_model(case, numeraire=("A", "labor"))
    b = solve_model(case, numeraire=("B", "labor"))
    dev = max(
        np.abs(a["counterfactual"]["x"] / b["counterfactual"]["x"] - 1).max(),
        np.abs(a["counterfactual"]["Y"] / b["counterfactual"]["Y"] - 1).max(),
        np.abs(
            a["counterfactual"]["real_income"] / a["benchmark"]["real_income"]
            - b["counterfactual"]["real_income"] / b["benchmark"]["real_income"]
        ).max(),
    )
    print(f"\n=== numeraire invariance ({case}) ===")
    print(f"max relative deviation of real allocations: {dev:.3e}")
    assert dev < 1e-6, "numeraire invariance failed"


if __name__ == "__main__":
    cases = sys.argv[1:] or ["3r"]
    for case in cases:
        compare(case, solve_model(case))
    numeraire_invariance("3r")
    print("\nAll checks passed.")
