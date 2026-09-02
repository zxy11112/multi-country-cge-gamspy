"""A small, solver-ready multi-country CGE model.

The model is deliberately compact:

* 2 sectors: food and manufacturing
* 2 primary factors: labor and capital
* constant-returns Cobb-Douglas production
* one Cobb-Douglas household per region
* one-level Armington aggregation across origins
* ad valorem import tariffs rebated to the destination household
* fixed regional factor endowments
* no intermediate inputs, saving, investment, or international factor mobility
* a fixed net-foreign-transfer closure for current accounts

Two artificial datasets are supplied:

1. ``2r``: A and B only, useful as a debugging model.
2. ``3r``: A, B, and an explicit Rest of World (ROW), useful for showing
   trade diversion.

ROW treatment in the 3-region case
----------------------------------
ROW is a full endogenous region, not a residual balancing account. It has its
own production, factor markets, household, prices, income, bilateral trade,
and welfare. Its factor endowments are fixed and its producer prices adjust.
The benchmark current account is fixed through a net foreign transfer; in the
supplied data that transfer is zero for every region. The ROW-manufacturing
commodity-market equation is omitted only to remove the Walras-law redundancy;
its residual is calculated and checked after every solve.

Run from the command line, for example:

    python multicountry_cge.py --case 3r --tariff-rate 0.10 \
        --output-dir outputs/three_region

Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


Array = np.ndarray


@dataclass(frozen=True)
class CGEData:
    """Benchmark data for the CGE model.

    Array dimensions
    ----------------
    bilateral_use0[origin, sector, destination]
        Benchmark physical quantity of each origin-sector variety used by each
        destination. Benchmark producer prices are normalized to one.

    factor_payments0[region, sector, factor]
        Benchmark factor payments. With benchmark factor prices equal to one,
        these are also benchmark factor quantities.

    sigma[sector]
        Armington elasticity across origins.

    net_foreign_transfer[region]
        Fixed net transfer received from abroad. Positive values finance a
        current-account deficit. Global transfers must sum to zero.

    baseline_tariffs[origin, sector, destination]
        Benchmark ad valorem tariff rates. Domestic rates must be zero.
    """

    regions: tuple[str, ...]
    sectors: tuple[str, ...]
    factors: tuple[str, ...]
    bilateral_use0: Array
    factor_payments0: Array
    sigma: Array
    net_foreign_transfer: Array
    baseline_tariffs: Array
    case_name: str


@dataclass
class CGESolution:
    """Solved equilibrium and diagnostics."""

    z: Array
    tariffs: Array
    producer_price: Array
    factor_price: Array
    output: Array
    income: Array
    purchaser_price: Array
    composite_price: Array
    composite_consumption: Array
    bilateral_use: Array
    factor_use: Array
    tariff_revenue: Array
    exports: Array
    imports: Array
    cost_of_living: Array
    real_income: Array
    equation_residuals: Array
    goods_market_log_residuals: Array
    solver_success: bool
    solver_message: str
    function_evaluations: int

    @property
    def max_equation_residual(self) -> float:
        return float(np.max(np.abs(self.equation_residuals)))

    @property
    def max_goods_market_residual(self) -> float:
        return float(np.max(np.abs(self.goods_market_log_residuals)))


class SimpleMultiCountryCGE:
    """Calibrate and solve the static multi-country model.

    The nonlinear system is written in log variables to preserve positivity.
    One factor price is fixed as the numeraire. One commodity-market equation
    is omitted because Walras' law makes it redundant; the omitted equation is
    nevertheless evaluated after the solve.
    """

    def __init__(
        self,
        data: CGEData,
        *,
        numeraire: tuple[str, str] = ("A", "labor"),
        dropped_market: tuple[str, str] | None = None,
    ) -> None:
        self.data = data
        self.regions = data.regions
        self.sectors = data.sectors
        self.factors = data.factors
        self.R = len(self.regions)
        self.S = len(self.sectors)
        self.F = len(self.factors)

        self.r_index = {name: idx for idx, name in enumerate(self.regions)}
        self.s_index = {name: idx for idx, name in enumerate(self.sectors)}
        self.f_index = {name: idx for idx, name in enumerate(self.factors)}

        if numeraire[0] not in self.r_index or numeraire[1] not in self.f_index:
            raise ValueError(f"Unknown numeraire {numeraire!r}.")
        self.numeraire = (
            self.r_index[numeraire[0]],
            self.f_index[numeraire[1]],
        )

        if dropped_market is None:
            dropped_market = (self.regions[-1], self.sectors[-1])
        if (
            dropped_market[0] not in self.r_index
            or dropped_market[1] not in self.s_index
        ):
            raise ValueError(f"Unknown dropped market {dropped_market!r}.")
        self.dropped_market = (
            self.r_index[dropped_market[0]],
            self.s_index[dropped_market[1]],
        )

        self._validate_and_calibrate()
        self._build_variable_index()

    # ------------------------------------------------------------------
    # Calibration and validation
    # ------------------------------------------------------------------
    def _validate_and_calibrate(self) -> None:
        d = self.data
        expected_x = (self.R, self.S, self.R)
        expected_fp = (self.R, self.S, self.F)
        expected_tau = (self.R, self.S, self.R)

        if d.bilateral_use0.shape != expected_x:
            raise ValueError(
                f"bilateral_use0 has shape {d.bilateral_use0.shape}; "
                f"expected {expected_x}."
            )
        if d.factor_payments0.shape != expected_fp:
            raise ValueError(
                f"factor_payments0 has shape {d.factor_payments0.shape}; "
                f"expected {expected_fp}."
            )
        if d.baseline_tariffs.shape != expected_tau:
            raise ValueError(
                f"baseline_tariffs has shape {d.baseline_tariffs.shape}; "
                f"expected {expected_tau}."
            )
        if d.sigma.shape != (self.S,):
            raise ValueError("sigma must have one entry per sector.")
        if d.net_foreign_transfer.shape != (self.R,):
            raise ValueError("net_foreign_transfer must have one entry per region.")
        if np.any(d.bilateral_use0 <= 0.0):
            raise ValueError(
                "All benchmark bilateral flows must be strictly positive in "
                "this simple CES implementation."
            )
        if np.any(d.factor_payments0 <= 0.0):
            raise ValueError("All benchmark factor payments must be positive.")
        if np.any(d.sigma <= 1.0):
            raise ValueError("This implementation requires Armington sigma > 1.")
        if np.any(d.baseline_tariffs <= -1.0):
            raise ValueError("Tariff rates must be greater than -1.")
        if not np.isclose(np.sum(d.net_foreign_transfer), 0.0, atol=1e-12):
            raise ValueError("Global net foreign transfers must sum to zero.")

        for r in range(self.R):
            if not np.allclose(d.baseline_tariffs[r, :, r], 0.0, atol=1e-14):
                raise ValueError("Domestic purchases cannot carry import tariffs.")

        # Benchmark producer and factor prices are normalized to one.
        self.p0 = np.ones((self.R, self.S))
        self.w0 = np.ones((self.R, self.F))
        self.P0 = np.ones((self.R, self.S))  # destination x sector

        self.x0 = np.asarray(d.bilateral_use0, dtype=float)
        self.factor_payments0 = np.asarray(d.factor_payments0, dtype=float)
        self.tau0 = np.asarray(d.baseline_tariffs, dtype=float)
        self.B = np.asarray(d.net_foreign_transfer, dtype=float)
        self.sigma = np.asarray(d.sigma, dtype=float)

        # Benchmark output of every origin-sector variety.
        self.Y0 = np.sum(self.x0, axis=2)

        # Production cost shares and regional factor endowments.
        sector_factor_cost = np.sum(self.factor_payments0, axis=2)
        if not np.allclose(sector_factor_cost, self.Y0, rtol=1e-11, atol=1e-11):
            raise ValueError(
                "For every region-sector, benchmark factor payments must sum "
                "to benchmark output value."
            )
        self.alpha = self.factor_payments0 / self.Y0[:, :, None]
        if not np.allclose(np.sum(self.alpha, axis=2), 1.0, atol=1e-12):
            raise ValueError("Production factor shares must sum to one.")
        self.factor_endowment = np.sum(self.factor_payments0, axis=1)

        # Benchmark purchaser prices and destination-sector expenditure.
        self.q0 = self.p0[:, :, None] * (1.0 + self.tau0)
        self.expenditure0 = np.einsum("oid,oid->di", self.q0, self.x0)

        # Tariff revenue is rebated to the destination household.
        self.tariff_revenue0 = np.einsum(
            "oid,oi,oid->d", self.tau0, self.p0, self.x0
        )
        factor_income0 = np.sum(self.factor_endowment, axis=1)
        self.I0 = factor_income0 + self.tariff_revenue0 + self.B

        if np.any(self.I0 <= 0.0):
            raise ValueError("Every benchmark household must have positive income.")
        if not np.allclose(
            np.sum(self.expenditure0, axis=1), self.I0, rtol=1e-11, atol=1e-11
        ):
            raise ValueError(
                "Benchmark household expenditure must equal factor income plus "
                "tariff revenue plus the fixed net foreign transfer."
            )

        # With P0 normalized to one, composite quantity equals expenditure.
        self.C0 = self.expenditure0.copy()
        self.beta = self.expenditure0 / self.I0[:, None]
        if not np.allclose(np.sum(self.beta, axis=1), 1.0, atol=1e-12):
            raise ValueError("Household budget shares must sum to one.")

        # Calibrated Armington expenditure shares.
        self.armington_share0 = np.empty_like(self.x0)
        for d_idx in range(self.R):
            for i_idx in range(self.S):
                self.armington_share0[:, i_idx, d_idx] = (
                    self.q0[:, i_idx, d_idx]
                    * self.x0[:, i_idx, d_idx]
                    / self.expenditure0[d_idx, i_idx]
                )
        if not np.allclose(
            np.sum(self.armington_share0, axis=0), 1.0, atol=1e-12
        ):
            raise ValueError("Armington source shares must sum to one.")

        # Check benchmark current-account accounting at producer prices.
        exports0, imports0 = self._trade_values(self.p0, self.x0)
        ca_residual0 = imports0 - exports0 - self.B
        if not np.allclose(ca_residual0, 0.0, rtol=1e-11, atol=1e-11):
            raise ValueError(
                "Benchmark external accounts do not satisfy M - X = B. "
                f"Residuals: {ca_residual0}."
            )

    def _build_variable_index(self) -> None:
        self.p_idx: dict[tuple[int, int], int] = {}
        self.w_idx: dict[tuple[int, int], int] = {}
        self.Y_idx: dict[tuple[int, int], int] = {}
        self.I_idx: dict[int, int] = {}

        k = 0
        for r in range(self.R):
            for i in range(self.S):
                self.p_idx[(r, i)] = k
                k += 1
        for r in range(self.R):
            for f in range(self.F):
                if (r, f) == self.numeraire:
                    continue
                self.w_idx[(r, f)] = k
                k += 1
        for r in range(self.R):
            for i in range(self.S):
                self.Y_idx[(r, i)] = k
                k += 1
        for r in range(self.R):
            self.I_idx[r] = k
            k += 1

        self.n_variables = k
        expected = 2 * self.R * self.S + self.R * self.F + self.R - 1
        if self.n_variables != expected:
            raise RuntimeError("Internal variable-count error.")

    # ------------------------------------------------------------------
    # Equilibrium system
    # ------------------------------------------------------------------
    def benchmark_initial_vector(self) -> Array:
        z = np.zeros(self.n_variables)
        for r in range(self.R):
            for i in range(self.S):
                z[self.p_idx[(r, i)]] = np.log(self.p0[r, i])
                z[self.Y_idx[(r, i)]] = np.log(self.Y0[r, i])
        for r in range(self.R):
            for f in range(self.F):
                if (r, f) != self.numeraire:
                    z[self.w_idx[(r, f)]] = np.log(self.w0[r, f])
        for r in range(self.R):
            z[self.I_idx[r]] = np.log(self.I0[r])
        return z

    def _unpack(self, z: Array) -> tuple[Array, Array, Array, Array]:
        p = np.empty((self.R, self.S))
        w = np.ones((self.R, self.F))
        Y = np.empty((self.R, self.S))
        I = np.empty(self.R)

        for r in range(self.R):
            for i in range(self.S):
                p[r, i] = np.exp(z[self.p_idx[(r, i)]])
                Y[r, i] = np.exp(z[self.Y_idx[(r, i)]])
        for r in range(self.R):
            for f in range(self.F):
                if (r, f) == self.numeraire:
                    w[r, f] = 1.0
                else:
                    w[r, f] = np.exp(z[self.w_idx[(r, f)]])
        for r in range(self.R):
            I[r] = np.exp(z[self.I_idx[r]])
        return p, w, Y, I

    def _demand_system(
        self, p: Array, I: Array, tariffs: Array
    ) -> tuple[Array, Array, Array, Array]:
        q = p[:, :, None] * (1.0 + tariffs)
        q_ratio = q / self.q0

        P = np.empty((self.R, self.S))  # destination x sector
        for d in range(self.R):
            for i in range(self.S):
                inside = np.sum(
                    self.armington_share0[:, i, d]
                    * q_ratio[:, i, d] ** (1.0 - self.sigma[i])
                )
                P[d, i] = inside ** (1.0 / (1.0 - self.sigma[i]))

        C = self.beta * I[:, None] / P

        x = np.empty_like(self.x0)
        for d in range(self.R):
            for i in range(self.S):
                x[:, i, d] = (
                    self.x0[:, i, d]
                    * (C[d, i] / self.C0[d, i])
                    * q_ratio[:, i, d] ** (-self.sigma[i])
                    * (P[d, i] / self.P0[d, i]) ** self.sigma[i]
                )
        return q, P, C, x

    def _residual(self, z: Array, tariffs: Array) -> Array:
        p, w, Y, I = self._unpack(z)
        _, _, _, x = self._demand_system(p, I, tariffs)
        residuals: list[float] = []

        # Zero-profit / unit-cost conditions.
        for r in range(self.R):
            for i in range(self.S):
                log_unit_cost = np.sum(self.alpha[r, i, :] * np.log(w[r, :]))
                residuals.append(np.log(p[r, i]) - log_unit_cost)

        # Regional factor-market clearing.
        for r in range(self.R):
            for f in range(self.F):
                factor_demand = np.sum(
                    self.alpha[r, :, f] * p[r, :] * Y[r, :] / w[r, f]
                )
                residuals.append(
                    np.log(factor_demand / self.factor_endowment[r, f])
                )

        # Global goods-market clearing, except one redundant equation.
        for r in range(self.R):
            for i in range(self.S):
                if (r, i) == self.dropped_market:
                    continue
                world_demand = np.sum(x[r, i, :])
                residuals.append(np.log(Y[r, i] / world_demand))

        # Household income: factor income + local tariff revenue + transfer.
        for d in range(self.R):
            factor_income = np.sum(w[d, :] * self.factor_endowment[d, :])
            tariff_revenue = np.sum(
                tariffs[:, :, d] * p[:, :] * x[:, :, d]
            )
            rhs = factor_income + tariff_revenue + self.B[d]
            if rhs <= 0.0 or not np.isfinite(rhs):
                return np.full(self.n_variables, 1.0e6)
            residuals.append(np.log(I[d] / rhs))

        result = np.asarray(residuals, dtype=float)
        if result.shape != (self.n_variables,):
            raise RuntimeError(
                f"Residual vector has shape {result.shape}; "
                f"expected {(self.n_variables,)}."
            )
        return result

    def solve(
        self,
        tariffs: Array | None = None,
        *,
        initial_solution: CGESolution | None = None,
        tolerance: float = 1e-10,
        max_nfev: int = 10_000,
    ) -> CGESolution:
        """Solve an equilibrium for the supplied tariff matrix."""

        if tariffs is None:
            tariffs = self.tau0.copy()
        tariffs = np.asarray(tariffs, dtype=float)
        if tariffs.shape != (self.R, self.S, self.R):
            raise ValueError("Tariff array has an incorrect shape.")
        if np.any(tariffs <= -1.0):
            raise ValueError("Every tariff rate must be greater than -1.")
        for r in range(self.R):
            if not np.allclose(tariffs[r, :, r], 0.0, atol=1e-14):
                raise ValueError("Domestic purchases cannot carry import tariffs.")

        z_start = (
            initial_solution.z.copy()
            if initial_solution is not None
            else self.benchmark_initial_vector()
        )

        result = least_squares(
            lambda z: self._residual(z, tariffs),
            z_start,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=max_nfev,
        )

        solution = self._assemble_solution(
            result.x,
            tariffs,
            result.fun,
            result.success,
            result.message,
            result.nfev,
        )

        max_residual = max(
            solution.max_equation_residual,
            solution.max_goods_market_residual,
        )
        if (not result.success) or max_residual > tolerance:
            raise RuntimeError(
                "CGE solve failed or did not reach the requested tolerance. "
                f"success={result.success}, max residual={max_residual:.3e}, "
                f"message={result.message}"
            )
        return solution

    def _assemble_solution(
        self,
        z: Array,
        tariffs: Array,
        equation_residuals: Array,
        solver_success: bool,
        solver_message: str,
        function_evaluations: int,
    ) -> CGESolution:
        p, w, Y, I = self._unpack(z)
        q, P, C, x = self._demand_system(p, I, tariffs)

        factor_use = np.empty((self.R, self.S, self.F))
        for r in range(self.R):
            for i in range(self.S):
                for f in range(self.F):
                    factor_use[r, i, f] = (
                        self.alpha[r, i, f] * p[r, i] * Y[r, i] / w[r, f]
                    )

        tariff_revenue = np.empty(self.R)
        for d in range(self.R):
            tariff_revenue[d] = np.sum(
                tariffs[:, :, d] * p[:, :] * x[:, :, d]
            )

        exports, imports = self._trade_values(p, x)
        cost_of_living = np.prod(P ** self.beta, axis=1)
        real_income = I / cost_of_living

        goods_market_log_residuals = np.empty((self.R, self.S))
        for r in range(self.R):
            for i in range(self.S):
                goods_market_log_residuals[r, i] = np.log(
                    Y[r, i] / np.sum(x[r, i, :])
                )

        return CGESolution(
            z=z.copy(),
            tariffs=tariffs.copy(),
            producer_price=p,
            factor_price=w,
            output=Y,
            income=I,
            purchaser_price=q,
            composite_price=P,
            composite_consumption=C,
            bilateral_use=x,
            factor_use=factor_use,
            tariff_revenue=tariff_revenue,
            exports=exports,
            imports=imports,
            cost_of_living=cost_of_living,
            real_income=real_income,
            equation_residuals=np.asarray(equation_residuals, dtype=float),
            goods_market_log_residuals=goods_market_log_residuals,
            solver_success=bool(solver_success),
            solver_message=str(solver_message),
            function_evaluations=int(function_evaluations),
        )

    def _trade_values(self, p: Array, x: Array) -> tuple[Array, Array]:
        exports = np.zeros(self.R)
        imports = np.zeros(self.R)
        for origin in range(self.R):
            for destination in range(self.R):
                if origin == destination:
                    continue
                value = np.sum(p[origin, :] * x[origin, :, destination])
                exports[origin] += value
                imports[destination] += value
        return exports, imports

    # ------------------------------------------------------------------
    # Diagnostics and reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _max_relative_error(actual: Array, target: Array) -> float:
        denominator = np.maximum(np.abs(target), 1.0)
        return float(np.max(np.abs(actual - target) / denominator))

    def benchmark_replication_report(self, solution: CGESolution) -> pd.DataFrame:
        rows = [
            ("producer prices", self._max_relative_error(solution.producer_price, self.p0)),
            ("factor prices", self._max_relative_error(solution.factor_price, self.w0)),
            ("outputs", self._max_relative_error(solution.output, self.Y0)),
            ("household income", self._max_relative_error(solution.income, self.I0)),
            ("composite prices", self._max_relative_error(solution.composite_price, self.P0)),
            ("composite consumption", self._max_relative_error(solution.composite_consumption, self.C0)),
            ("bilateral quantities", self._max_relative_error(solution.bilateral_use, self.x0)),
            ("all goods markets", solution.max_goods_market_residual),
            ("solved equations", solution.max_equation_residual),
        ]
        return pd.DataFrame(rows, columns=["check", "maximum_absolute_or_relative_error"])

    def assert_benchmark_replication(
        self, solution: CGESolution, tolerance: float = 1e-9
    ) -> None:
        report = self.benchmark_replication_report(solution)
        worst = float(report["maximum_absolute_or_relative_error"].max())
        if worst > tolerance:
            raise AssertionError(
                f"Benchmark replication failed: worst error {worst:.3e}."
            )

    def country_summary(
        self, benchmark: CGESolution, counterfactual: CGESolution
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for r, name in enumerate(self.regions):
            real_ratio = counterfactual.real_income[r] / benchmark.real_income[r]
            external_residual = (
                counterfactual.imports[r]
                - counterfactual.exports[r]
                - self.B[r]
            )
            rows.append(
                {
                    "region": name,
                    "benchmark_income": benchmark.income[r],
                    "counterfactual_income": counterfactual.income[r],
                    "nominal_income_pct_change": 100.0
                    * (counterfactual.income[r] / benchmark.income[r] - 1.0),
                    "cost_of_living_pct_change": 100.0
                    * (
                        counterfactual.cost_of_living[r]
                        / benchmark.cost_of_living[r]
                        - 1.0
                    ),
                    "welfare_pct_change": 100.0 * (real_ratio - 1.0),
                    "equivalent_variation": benchmark.income[r]
                    * (real_ratio - 1.0),
                    "tariff_revenue": counterfactual.tariff_revenue[r],
                    "exports_at_producer_prices": counterfactual.exports[r],
                    "imports_at_producer_prices": counterfactual.imports[r],
                    "fixed_net_foreign_transfer": self.B[r],
                    "external_balance_residual_M_minus_X_minus_B": external_residual,
                }
            )
        return pd.DataFrame(rows)

    def sector_summary(
        self, benchmark: CGESolution, counterfactual: CGESolution
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for r, region in enumerate(self.regions):
            for i, sector in enumerate(self.sectors):
                rows.append(
                    {
                        "region": region,
                        "sector": sector,
                        "benchmark_output": benchmark.output[r, i],
                        "counterfactual_output": counterfactual.output[r, i],
                        "output_pct_change": 100.0
                        * (counterfactual.output[r, i] / benchmark.output[r, i] - 1.0),
                        "producer_price_pct_change": 100.0
                        * (
                            counterfactual.producer_price[r, i]
                            / benchmark.producer_price[r, i]
                            - 1.0
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def factor_summary(
        self, benchmark: CGESolution, counterfactual: CGESolution
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for r, region in enumerate(self.regions):
            for f, factor in enumerate(self.factors):
                rows.append(
                    {
                        "region": region,
                        "factor": factor,
                        "benchmark_factor_price": benchmark.factor_price[r, f],
                        "counterfactual_factor_price": counterfactual.factor_price[r, f],
                        "factor_price_pct_change": 100.0
                        * (
                            counterfactual.factor_price[r, f]
                            / benchmark.factor_price[r, f]
                            - 1.0
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def bilateral_trade_summary(
        self, benchmark: CGESolution, counterfactual: CGESolution
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for origin, origin_name in enumerate(self.regions):
            for destination, destination_name in enumerate(self.regions):
                for i, sector in enumerate(self.sectors):
                    x0 = benchmark.bilateral_use[origin, i, destination]
                    x1 = counterfactual.bilateral_use[origin, i, destination]
                    rows.append(
                        {
                            "origin": origin_name,
                            "destination": destination_name,
                            "sector": sector,
                            "is_international": origin != destination,
                            "benchmark_quantity": x0,
                            "counterfactual_quantity": x1,
                            "quantity_pct_change": 100.0 * (x1 / x0 - 1.0),
                            "benchmark_purchaser_price": benchmark.purchaser_price[
                                origin, i, destination
                            ],
                            "counterfactual_purchaser_price": counterfactual.purchaser_price[
                                origin, i, destination
                            ],
                            "benchmark_tariff_rate": benchmark.tariffs[
                                origin, i, destination
                            ],
                            "counterfactual_tariff_rate": counterfactual.tariffs[
                                origin, i, destination
                            ],
                        }
                    )
        return pd.DataFrame(rows)

    def market_residual_summary(self, solution: CGESolution) -> pd.DataFrame:
        rows: list[dict[str, float | str | bool]] = []
        for r, region in enumerate(self.regions):
            for i, sector in enumerate(self.sectors):
                rows.append(
                    {
                        "region": region,
                        "sector": sector,
                        "was_omitted_from_solver": (r, i) == self.dropped_market,
                        "log_supply_demand_ratio": solution.goods_market_log_residuals[
                            r, i
                        ],
                    }
                )
        return pd.DataFrame(rows)

    def metadata(
        self,
        benchmark: CGESolution,
        counterfactual: CGESolution,
        shock: Mapping[str, object],
    ) -> dict[str, object]:
        dropped_region = self.regions[self.dropped_market[0]]
        dropped_sector = self.sectors[self.dropped_market[1]]
        numeraire_region = self.regions[self.numeraire[0]]
        numeraire_factor = self.factors[self.numeraire[1]]
        has_row = "ROW" in self.regions

        return {
            "case_name": self.data.case_name,
            "regions": list(self.regions),
            "sectors": list(self.sectors),
            "factors": list(self.factors),
            "shock": dict(shock),
            "closure": {
                "numeraire": f"{numeraire_region}:{numeraire_factor}=1",
                "factor_endowments": "fixed by region",
                "factor_mobility": "within-region across sectors only",
                "international_factor_mobility": "none",
                "household_saving": "none; households spend all income",
                "government": "tariff revenue rebated lump-sum locally",
                "current_account": "fixed net foreign transfer B; M-X=B",
                "net_foreign_transfer": {
                    region: float(self.B[idx])
                    for idx, region in enumerate(self.regions)
                },
                "omitted_equation_for_walras_law": (
                    f"{dropped_region}:{dropped_sector} goods market"
                ),
                "omitted_equation_residual_is_checked": True,
            },
            "row_treatment": (
                {
                    "implemented": True,
                    "type": "explicit endogenous aggregate region",
                    "full_production_system": True,
                    "full_factor_markets": True,
                    "representative_household": True,
                    "producer_prices": "endogenous",
                    "factor_prices": "endogenous",
                    "trade_flows": "bilateral Armington flows with A and B",
                    "within_ROW_trade": (
                        "represented as ROW purchases from ROW; no internal "
                        "country detail"
                    ),
                    "policy_scope": (
                        "the sample tariff applies only to A imports from B; "
                        "A imports from ROW are initially untaxed"
                    ),
                    "interpretation_limit": (
                        "ROW welfare is an aggregate; distribution across the "
                        "countries inside ROW is not identified"
                    ),
                    "not_a_balancing_account": True,
                }
                if has_row
                else {
                    "implemented": False,
                    "note": "This is the two-region debugging case.",
                }
            ),
            "solver": {
                "algorithm": "scipy.optimize.least_squares on log variables",
                "benchmark_function_evaluations": benchmark.function_evaluations,
                "counterfactual_function_evaluations": counterfactual.function_evaluations,
                "benchmark_max_equation_residual": benchmark.max_equation_residual,
                "benchmark_max_goods_market_residual": benchmark.max_goods_market_residual,
                "counterfactual_max_equation_residual": counterfactual.max_equation_residual,
                "counterfactual_max_goods_market_residual": counterfactual.max_goods_market_residual,
            },
        }

    def write_outputs(
        self,
        output_dir: str | Path,
        benchmark: CGESolution,
        counterfactual: CGESolution,
        shock: Mapping[str, object],
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.benchmark_replication_report(benchmark).to_csv(
            output_path / "benchmark_replication.csv", index=False
        )
        self.country_summary(benchmark, counterfactual).to_csv(
            output_path / "country_summary.csv", index=False
        )
        self.sector_summary(benchmark, counterfactual).to_csv(
            output_path / "sector_summary.csv", index=False
        )
        self.factor_summary(benchmark, counterfactual).to_csv(
            output_path / "factor_summary.csv", index=False
        )
        self.bilateral_trade_summary(benchmark, counterfactual).to_csv(
            output_path / "bilateral_trade_summary.csv", index=False
        )
        self.market_residual_summary(counterfactual).to_csv(
            output_path / "market_residuals.csv", index=False
        )
        with (output_path / "model_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(
                self.metadata(benchmark, counterfactual, shock),
                fh,
                indent=2,
                ensure_ascii=False,
            )


# ----------------------------------------------------------------------
# Artificial benchmark datasets
# ----------------------------------------------------------------------
def build_two_region_debug_data() -> CGEData:
    """Return the 2-country, 2-sector benchmark from the guideline."""

    regions = ("A", "B")
    sectors = ("food", "manufacturing")
    factors = ("labor", "capital")

    # bilateral_use0[origin, sector, destination]
    bilateral_use0 = np.empty((2, 2, 2), dtype=float)
    bilateral_use0[:, 0, :] = np.array(
        [
            [45.0, 15.0],  # A-food used by A and B
            [15.0, 15.0],  # B-food used by A and B
        ]
    )
    bilateral_use0[:, 1, :] = np.array(
        [
            [25.0, 15.0],  # A-manufacturing
            [15.0, 55.0],  # B-manufacturing
        ]
    )

    factor_payments0 = np.empty((2, 2, 2), dtype=float)
    factor_payments0[0, 0, :] = [42.0, 18.0]
    factor_payments0[0, 1, :] = [20.0, 20.0]
    factor_payments0[1, 0, :] = [18.0, 12.0]
    factor_payments0[1, 1, :] = [28.0, 42.0]

    return CGEData(
        regions=regions,
        sectors=sectors,
        factors=factors,
        bilateral_use0=bilateral_use0,
        factor_payments0=factor_payments0,
        sigma=np.array([2.0, 4.0]),
        net_foreign_transfer=np.zeros(2),
        baseline_tariffs=np.zeros((2, 2, 2)),
        case_name="two_region_debug",
    )


def build_three_region_row_data() -> CGEData:
    """Return the default A-B-ROW benchmark.

    Every origin-destination-sector flow is positive. A and B retain the same
    total outputs and factor payments as in the two-region benchmark. ROW has
    output 90 in food and 110 in manufacturing. Each region has balanced trade
    in the benchmark, so all fixed net foreign transfers are zero.
    """

    regions = ("A", "B", "ROW")
    sectors = ("food", "manufacturing")
    factors = ("labor", "capital")

    bilateral_use0 = np.empty((3, 2, 3), dtype=float)

    # Rows are origins A, B, ROW; columns are destinations A, B, ROW.
    bilateral_use0[:, 0, :] = np.array(
        [
            [42.0, 8.0, 10.0],
            [8.0, 14.0, 8.0],
            [10.0, 8.0, 72.0],
        ]
    )
    bilateral_use0[:, 1, :] = np.array(
        [
            [24.0, 8.0, 8.0],
            [8.0, 50.0, 12.0],
            [8.0, 12.0, 90.0],
        ]
    )

    factor_payments0 = np.empty((3, 2, 2), dtype=float)
    factor_payments0[0, 0, :] = [42.0, 18.0]
    factor_payments0[0, 1, :] = [20.0, 20.0]
    factor_payments0[1, 0, :] = [18.0, 12.0]
    factor_payments0[1, 1, :] = [28.0, 42.0]
    factor_payments0[2, 0, :] = [63.0, 27.0]
    factor_payments0[2, 1, :] = [44.0, 66.0]

    return CGEData(
        regions=regions,
        sectors=sectors,
        factors=factors,
        bilateral_use0=bilateral_use0,
        factor_payments0=factor_payments0,
        sigma=np.array([2.0, 4.0]),
        net_foreign_transfer=np.zeros(3),
        baseline_tariffs=np.zeros((3, 2, 3)),
        case_name="three_region_explicit_ROW",
    )


def build_model(
    case: str,
    *,
    numeraire: tuple[str, str] = ("A", "labor"),
) -> SimpleMultiCountryCGE:
    if case == "2r":
        data = build_two_region_debug_data()
        dropped_market = ("B", "manufacturing")
    elif case == "3r":
        data = build_three_region_row_data()
        dropped_market = ("ROW", "manufacturing")
    else:
        raise ValueError("case must be '2r' or '3r'.")
    return SimpleMultiCountryCGE(
        data,
        numeraire=numeraire,
        dropped_market=dropped_market,
    )


def make_tariff_shock(
    model: SimpleMultiCountryCGE,
    *,
    origin: str,
    sector: str,
    destination: str,
    rate: float,
) -> Array:
    if origin == destination:
        raise ValueError("The example shock must apply to an international flow.")
    tariffs = model.tau0.copy()
    tariffs[
        model.r_index[origin],
        model.s_index[sector],
        model.r_index[destination],
    ] = rate
    return tariffs


def _format_console_report(
    model: SimpleMultiCountryCGE,
    benchmark: CGESolution,
    counterfactual: CGESolution,
    *,
    origin: str,
    sector: str,
    destination: str,
    rate: float,
    output_dir: Path,
) -> str:
    country = model.country_summary(benchmark, counterfactual)
    bilateral = model.bilateral_trade_summary(benchmark, counterfactual)
    destination_rows = bilateral[
        (bilateral["destination"] == destination)
        & (bilateral["sector"] == sector)
    ][["origin", "quantity_pct_change"]]

    lines = [
        f"Case: {model.data.case_name}",
        f"Shock: {destination} tariff on {origin} {sector} = {100.0 * rate:.2f}%",
        (
            "Numeraire: "
            f"{model.regions[model.numeraire[0]]}:"
            f"{model.factors[model.numeraire[1]]}=1"
        ),
        (
            "Omitted Walras-law equation: "
            f"{model.regions[model.dropped_market[0]]}:"
            f"{model.sectors[model.dropped_market[1]]} goods market"
        ),
        f"Benchmark max residual: {max(benchmark.max_equation_residual, benchmark.max_goods_market_residual):.3e}",
        f"Counterfactual max residual: {max(counterfactual.max_equation_residual, counterfactual.max_goods_market_residual):.3e}",
        "",
        "Welfare change (%):",
    ]
    for row in country.itertuples(index=False):
        lines.append(f"  {row.region:>4}: {row.welfare_pct_change: .6f}")

    lines.extend(
        [
            "",
            f"{destination} demand for {sector} by origin (% change):",
        ]
    )
    for row in destination_rows.itertuples(index=False):
        lines.append(f"  {row.origin:>4}: {row.quantity_pct_change: .6f}")

    lines.extend(["", f"Output files: {output_dir.resolve()}"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("2r", "3r"),
        default="3r",
        help="2-region debugging case or 3-region explicit-ROW case.",
    )
    parser.add_argument("--tariff-rate", type=float, default=0.10)
    parser.add_argument("--shock-origin", default="B")
    parser.add_argument("--shock-sector", default="manufacturing")
    parser.add_argument("--shock-destination", default="A")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    args = parser.parse_args(argv)

    model = build_model(args.case)
    benchmark = model.solve(model.tau0)
    model.assert_benchmark_replication(benchmark)

    tariffs = make_tariff_shock(
        model,
        origin=args.shock_origin,
        sector=args.shock_sector,
        destination=args.shock_destination,
        rate=args.tariff_rate,
    )
    counterfactual = model.solve(tariffs, initial_solution=benchmark)

    shock = {
        "destination": args.shock_destination,
        "origin": args.shock_origin,
        "sector": args.shock_sector,
        "ad_valorem_rate": args.tariff_rate,
    }
    model.write_outputs(args.output_dir, benchmark, counterfactual, shock)

    print(
        _format_console_report(
            model,
            benchmark,
            counterfactual,
            origin=args.shock_origin,
            sector=args.shock_sector,
            destination=args.shock_destination,
            rate=args.tariff_rate,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
