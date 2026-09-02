"""Regression tests for the small multi-country CGE model."""

from __future__ import annotations

import numpy as np

from multicountry_cge import build_model, make_tariff_shock


def _solve_case(case: str):
    model = build_model(case)
    benchmark = model.solve(model.tau0)
    model.assert_benchmark_replication(benchmark, tolerance=1e-10)
    shock = make_tariff_shock(
        model,
        origin="B",
        sector="manufacturing",
        destination="A",
        rate=0.10,
    )
    counterfactual = model.solve(shock, initial_solution=benchmark)
    return model, benchmark, counterfactual


def test_two_region_benchmark_and_tariff_solve() -> None:
    model, benchmark, counterfactual = _solve_case("2r")
    assert benchmark.max_equation_residual < 1e-10
    assert counterfactual.max_goods_market_residual < 1e-10

    b = model.r_index["B"]
    a = model.r_index["A"]
    m = model.s_index["manufacturing"]
    assert counterfactual.bilateral_use[b, m, a] < benchmark.bilateral_use[b, m, a]


def test_three_region_benchmark_replication() -> None:
    model = build_model("3r")
    benchmark = model.solve(model.tau0)
    report = model.benchmark_replication_report(benchmark)
    assert report["maximum_absolute_or_relative_error"].max() < 1e-10


def test_explicit_row_generates_trade_diversion() -> None:
    model, benchmark, counterfactual = _solve_case("3r")
    a = model.r_index["A"]
    b = model.r_index["B"]
    row = model.r_index["ROW"]
    m = model.s_index["manufacturing"]

    # A's targeted imports from B fall.
    assert counterfactual.bilateral_use[b, m, a] < benchmark.bilateral_use[b, m, a]
    # A substitutes toward the untaxed ROW source.
    assert counterfactual.bilateral_use[row, m, a] > benchmark.bilateral_use[row, m, a]
    # A also substitutes toward its domestic variety.
    assert counterfactual.bilateral_use[a, m, a] > benchmark.bilateral_use[a, m, a]


def test_current_accounts_and_omitted_market_clear() -> None:
    model, _, counterfactual = _solve_case("3r")
    external_residual = (
        counterfactual.imports - counterfactual.exports - model.B
    )
    assert np.max(np.abs(external_residual)) < 1e-10
    assert counterfactual.max_goods_market_residual < 1e-10


def test_numeraire_invariance_of_real_allocations() -> None:
    model_a = build_model("3r", numeraire=("A", "labor"))
    base_a = model_a.solve(model_a.tau0)
    shock_a = make_tariff_shock(
        model_a,
        origin="B",
        sector="manufacturing",
        destination="A",
        rate=0.10,
    )
    cf_a = model_a.solve(shock_a, initial_solution=base_a)

    model_b = build_model("3r", numeraire=("B", "labor"))
    base_b = model_b.solve(model_b.tau0)
    shock_b = make_tariff_shock(
        model_b,
        origin="B",
        sector="manufacturing",
        destination="A",
        rate=0.10,
    )
    cf_b = model_b.solve(shock_b, initial_solution=base_b)

    np.testing.assert_allclose(cf_a.output, cf_b.output, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        cf_a.bilateral_use, cf_b.bilateral_use, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        cf_a.composite_consumption,
        cf_b.composite_consumption,
        rtol=1e-9,
        atol=1e-9,
    )

    welfare_ratio_a = cf_a.real_income / base_a.real_income
    welfare_ratio_b = cf_b.real_income / base_b.real_income
    np.testing.assert_allclose(
        welfare_ratio_a, welfare_ratio_b, rtol=1e-9, atol=1e-9
    )
