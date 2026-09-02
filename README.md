# Simple Multi-Country CGE: A–B–ROW Python Prototype

This package implements a small static computable general equilibrium model that is simple enough to inspect equation by equation, but complete enough to demonstrate:

- bilateral trade by origin and destination;
- Armington substitution;
- sectoral production and factor reallocation;
- tariff revenue and household welfare;
- current-account accounting;
- trade diversion through an explicit Rest of World region;
- benchmark replication, Walras-law, and numeraire-invariance tests.

The default model is:

\[
3\text{ regions}\times 2\text{ sectors}\times 2\text{ factors},
\]

where the regions are \(A\), \(B\), and \(ROW\); the sectors are food and manufacturing; and the factors are labor and capital.

A two-region debugging version is also included. It reproduces the original \(A\)-\(B\) benchmark without ROW.

---

## 1. Files

- `multicountry_cge.py`: model, artificial datasets, solver, reporting, and command-line interface.
- `test_multicountry_cge.py`: five regression tests.
- `requirements.txt`: Python dependencies.
- `outputs/two_region/`: pre-generated results for the two-region debugging case.
- `outputs/three_region/`: pre-generated results for the explicit-ROW case.

The program writes:

- `benchmark_replication.csv`;
- `country_summary.csv`;
- `sector_summary.csv`;
- `factor_summary.csv`;
- `bilateral_trade_summary.csv`;
- `market_residuals.csv`;
- `model_metadata.json`.

---

## 2. Installation and execution

From the package directory:

```bash
python -m pip install -r requirements.txt
```

Run the default three-region case:

```bash
python multicountry_cge.py \
  --case 3r \
  --tariff-rate 0.10 \
  --output-dir outputs/three_region
```

Run the original two-region debugging model:

```bash
python multicountry_cge.py \
  --case 2r \
  --tariff-rate 0.10 \
  --output-dir outputs/two_region
```

Run the tests:

```bash
pytest -q
```

The supplied version passes all five tests.

---

# 3. Economic structure

Let:

- \(r\) denote the producing region;
- \(s\) denote the consuming region;
- \(i\) denote the sector;
- \(f\) denote the factor.

The bilateral quantity

\[
x_{ris}
\]

is the amount of sector \(i\) produced in origin \(r\) and used in destination \(s\).

## 3.1 Production

Production is constant-returns Cobb–Douglas:

\[
Y_{ri}
=
A_{ri}\prod_f F_{rif}^{\alpha_{rif}},
\qquad
\sum_f\alpha_{rif}=1.
\]

Factors are mobile across sectors inside their home region, but cannot move internationally.

Perfect competition gives the normalized unit-cost condition:

\[
\frac{p_{ri}}{p^0_{ri}}
=
\prod_f
\left(
\frac{w_{rf}}{w^0_{rf}}
\right)^{\alpha_{rif}}.
\]

Benchmark producer and factor prices are normalized to one.

Conditional factor demand is:

\[
F_{rif}
=
\alpha_{rif}\frac{p_{ri}Y_{ri}}{w_{rf}}.
\]

Regional factor markets clear:

\[
\sum_i F_{rif}=\bar F_{rf}.
\]

## 3.2 Tariff-inclusive bilateral prices

The purchaser price paid by destination \(s\) is:

\[
q_{ris}
=
(1+\tau_{ris})p_{ri}.
\]

Domestic tariff rates satisfy:

\[
\tau_{sis}=0.
\]

The sample counterfactual is:

\[
\tau_{B,\mathrm{manufacturing},A}:0\longrightarrow 0.10.
\]

Thus only country \(A\)'s imports of manufacturing from \(B\) receive the new tariff. Imports from ROW remain untaxed.

## 3.3 Armington aggregation

Each destination-sector composite is a CES aggregate over origins. The implementation uses calibrated expenditure shares:

\[
s^0_{ris}
=
\frac{q^0_{ris}x^0_{ris}}
{\sum_k q^0_{kis}x^0_{kis}}.
\]

The normalized composite price is:

\[
\frac{P_{si}}{P^0_{si}}
=
\left[
\sum_r s^0_{ris}
\left(
\frac{q_{ris}}{q^0_{ris}}
\right)^{1-\sigma_i}
\right]^{\frac{1}{1-\sigma_i}}.
\]

Bilateral demand is:

\[
\frac{x_{ris}}{x^0_{ris}}
=
\frac{C_{si}}{C^0_{si}}
\left(
\frac{q_{ris}}{q^0_{ris}}
\right)^{-\sigma_i}
\left(
\frac{P_{si}}{P^0_{si}}
\right)^{\sigma_i}.
\]

The artificial elasticities are:

\[
\sigma_{\mathrm{food}}=2,
\qquad
\sigma_{\mathrm{manufacturing}}=4.
\]

They are illustrative assumptions, not empirical estimates.

## 3.4 Household demand

Each region has one representative Cobb–Douglas household:

\[
U_s=\prod_i C_{si}^{\beta_{si}},
\qquad
\sum_i\beta_{si}=1.
\]

Demand is:

\[
C_{si}=\frac{\beta_{si}I_s}{P_{si}}.
\]

The household spends all current income. There is no saving or investment in this prototype.

## 3.5 Income and tariff revenue

Regional household income is:

\[
I_s
=
\sum_f w_{sf}\bar F_{sf}
+T_s+B_s,
\]

where \(B_s\) is the fixed net foreign transfer received by region \(s\), and:

\[
T_s
=
\sum_i\sum_{r\ne s}
\tau_{ris}p_{ri}x_{ris}.
\]

Tariff revenue is rebated lump-sum to the household in the destination region.

## 3.6 Goods markets

Every origin-sector variety clears globally:

\[
Y_{ri}=\sum_s x_{ris}.
\]

## 3.7 Current-account closure

Imports and exports are valued at producer prices:

\[
M_s
=
\sum_i\sum_{r\ne s}p_{ri}x_{ris},
\]

\[
X_s
=
\sum_i\sum_{d\ne s}p_{si}x_{sid}.
\]

The closure is:

\[
M_s-X_s=B_s,
\qquad
\sum_s B_s=0.
\]

In both supplied benchmarks:

\[
B_A=B_B=B_{ROW}=0.
\]

Hence every region has balanced trade at producer prices in the benchmark and counterfactual:

\[
M_s=X_s.
\]

For an empirical model, \(B_s\) can be calibrated to benchmark current-account imbalances and held fixed in real or numeraire units. Global consistency still requires:

\[
\sum_s B_s=0.
\]

---

# 4. Exact treatment of Rest of World

## 4.1 Treatment implemented in this artifact

ROW is modeled as an **explicit endogenous aggregate economy**.

It is not:

- an infinitely elastic import supply curve;
- a fixed world price;
- a residual market-clearing account;
- a repository for accounting errors;
- a passive source of imports.

ROW has the same economic blocks as \(A\) and \(B\):

| Component | ROW treatment |
|---|---|
| Production | Two endogenous sectoral outputs |
| Factors | Fixed ROW labor and capital endowments |
| Factor allocation | Mobile across ROW sectors |
| International factor mobility | None |
| Producer prices | Endogenous |
| Factor prices | Endogenous |
| Household | One representative ROW household |
| Preferences | Cobb–Douglas across food and manufacturing |
| Imports | Armington demand for varieties from A and B |
| Exports | Endogenous supply to A and B through market clearing |
| Tariff revenue | Rebate to ROW household if ROW imposes a tariff |
| Current account | Fixed through \(B_{ROW}\); zero in the example |
| Welfare | Computed for the aggregate ROW household |

This treatment allows the tariff on \(A\)'s imports from \(B\) to generate genuine trade diversion:

\[
B\rightarrow A\text{ manufacturing falls},
\]

while:

\[
ROW\rightarrow A\text{ manufacturing rises}.
\]

ROW production, wages, capital returns, income, and welfare also adjust.

## 4.2 What “ROW domestic trade” means

The flow:

\[
x_{ROW,i,ROW}
\]

is treated as domestic absorption inside the aggregated ROW region.

It combines all transactions that remain internal after many countries are aggregated into ROW. Consequently, it includes trade between individual countries that are hidden inside the ROW aggregate. The model does not identify those internal bilateral flows.

## 4.3 Interpretation limit

ROW welfare is the welfare of one synthetic representative household. It does not reveal:

- which individual ROW countries gain or lose;
- distribution across households inside ROW;
- trade diversion among countries inside ROW;
- changes in internal ROW exchange rates or tariffs.

Those questions require disaggregating ROW into additional regions.

## 4.4 The omitted ROW equation is not a residual closure

The solver omits the equation:

\[
Y_{ROW,\mathrm{manufacturing}}
=
\sum_s x_{ROW,\mathrm{manufacturing},s}
\]

from the nonlinear system solely because one market equation is redundant under Walras' law after a numeraire is fixed.

The code evaluates this equation after the solve and requires its residual to be below the numerical tolerance. In the supplied tariff experiment, the maximum residual across all goods markets, including the omitted one, is approximately:

\[
6.7\times10^{-16}.
\]

Thus ROW does not absorb disequilibrium.

## 4.5 Alternative ROW closure not implemented here

A small-open-economy model might instead take ROW prices as fixed:

\[
p_{ROW,i}=\bar p_i.
\]

That treatment makes ROW an infinitely elastic source and destination at fixed world prices. To implement it correctly, one would remove the ROW production, factor-market, household, and welfare blocks and replace them with external import-supply/export-demand relationships and a foreign-exchange closure.

That is a different model. It is useful when \(A\) and \(B\) are genuinely small relative to world markets, but it suppresses feedback from the policy shock to ROW prices and production. The present artifact deliberately uses the endogenous-ROW treatment.

---

# 5. Artificial benchmark data

## 5.1 Two-region debugging case

Food flows, with origins in rows and destinations in columns:

| Origin | A | B | Output |
|---|---:|---:|---:|
| A | 45 | 15 | 60 |
| B | 15 | 15 | 30 |

Manufacturing flows:

| Origin | A | B | Output |
|---|---:|---:|---:|
| A | 25 | 15 | 40 |
| B | 15 | 55 | 70 |

## 5.2 Three-region explicit-ROW case

Food flows:

| Origin | A | B | ROW | Output |
|---|---:|---:|---:|---:|
| A | 42 | 8 | 10 | 60 |
| B | 8 | 14 | 8 | 30 |
| ROW | 10 | 8 | 72 | 90 |

Manufacturing flows:

| Origin | A | B | ROW | Output |
|---|---:|---:|---:|---:|
| A | 24 | 8 | 8 | 40 |
| B | 8 | 50 | 12 | 70 |
| ROW | 8 | 12 | 90 | 110 |

Destination absorption is:

| Destination | Food | Manufacturing | Total |
|---|---:|---:|---:|
| A | 60 | 40 | 100 |
| B | 30 | 70 | 100 |
| ROW | 90 | 110 | 200 |

The benchmark is constructed so that each region's producer-price imports equal its exports.

## 5.3 Factor payments

| Region-sector | Labor | Capital | Output value |
|---|---:|---:|---:|
| A-food | 42 | 18 | 60 |
| A-manufacturing | 20 | 20 | 40 |
| B-food | 18 | 12 | 30 |
| B-manufacturing | 28 | 42 | 70 |
| ROW-food | 63 | 27 | 90 |
| ROW-manufacturing | 44 | 66 | 110 |

This implies endowments:

\[
(\bar L_A,\bar K_A)=(62,38),
\]

\[
(\bar L_B,\bar K_B)=(46,54),
\]

\[
(\bar L_{ROW},\bar K_{ROW})=(107,93).
\]

---

# 6. Solver formulation

All price, output, and income variables are represented in logarithms. This guarantees positivity.

The unknowns are:

\[
\{p_{ri},w_{rf},Y_{ri},I_r\},
\]

except that one factor price is fixed:

\[
w_{A,\mathrm{labor}}=1.
\]

The system contains:

- all unit-cost conditions;
- all factor-market conditions;
- all but one goods-market condition;
- all household-income conditions.

The script uses:

```text
scipy.optimize.least_squares
```

on the square nonlinear system. Although the numerical routine is a least-squares solver, the accepted solution is required to have every equation residual below the specified equilibrium tolerance.

---

# 7. Benchmark replication

Before applying a policy shock, the program solves at the benchmark tariffs and checks that it reproduces:

\[
p=p^0,
\quad
w=w^0,
\quad
Y=Y^0,
\quad
I=I^0,
\quad
C=C^0,
\quad
x=x^0.
\]

It also verifies every goods market, including the market omitted from the solver.

For the supplied three-region case, the maximum benchmark residual is approximately:

\[
6.7\times10^{-16}.
\]

---

# 8. Illustrative tariff result

The experiment sets:

\[
\tau_{B,\mathrm{manufacturing},A}=10\%.
\]

In the explicit-ROW case, country \(A\)'s manufacturing demand changes by origin as follows:

| Origin | Percentage change in A's quantity demand |
|---|---:|
| A | +4.3234% |
| B | −22.8735% |
| ROW | +7.5537% |

This is the intended trade-diversion mechanism. A substitutes away from the tariffed \(B\) variety toward both its domestic variety and the untaxed ROW variety.

Illustrative welfare changes are:

| Region | Welfare change | Equivalent variation |
|---|---:|---:|
| A | +0.3349% | +0.3349 |
| B | −0.5329% | −0.5329 |
| ROW | +0.0541% | +0.1081 |

These values are properties of the artificial calibration and should not be interpreted as empirical tariff estimates.

The two-country model gives a larger gain for \(A\), partly because it prevents diversion toward a third-country supplier. That comparison illustrates why a bilateral empirical model normally needs ROW.

---

# 9. Welfare measure

The normalized Cobb–Douglas cost-of-living index is:

\[
\Pi_s=\prod_i P_{si}^{\beta_{si}}.
\]

Real income is:

\[
R_s=\frac{I_s}{\Pi_s}.
\]

The reported welfare change is:

\[
100\left(\frac{R_s^1}{R_s^0}-1\right).
\]

Equivalent variation in benchmark-income units is:

\[
EV_s
=
I_s^0
\left(
\frac{R_s^1}{R_s^0}-1
\right).
\]

---

# 10. Automated tests

The test suite verifies:

1. the two-region benchmark and tariff counterfactual solve;
2. the three-region benchmark replicates exactly;
3. the tariff reduces \(B\)-to-\(A\) manufacturing and increases ROW-to-\(A\) manufacturing;
4. all current accounts and the omitted goods market clear;
5. changing the numeraire leaves real allocations and welfare unchanged.

The fifth test is particularly important. It solves the same tariff experiment once with:

\[
w_{A,L}=1
\]

and once with:

\[
w_{B,L}=1.
\]

Nominal price levels differ, but quantities and welfare ratios must coincide.

---

# 11. How to replace the artificial data

The minimum empirical tables are:

## Bilateral quantities or consistently valued flows

```text
origin, destination, sector, benchmark_flow
```

Domestic use must be included as origin equal to destination.

## Factor payments

```text
region, sector, factor, benchmark_payment
```

## Tariffs

```text
origin, destination, sector, benchmark_rate
```

## Armington elasticities

```text
sector, elasticity
```

## Net foreign transfers

```text
region, net_transfer_received
```

Before calibration, reconcile:

\[
\text{sector output value}
=
\text{factor payments}
\]

in this no-intermediate-input model;

\[
Y^0_{ri}=\sum_s x^0_{ris};
\]

\[
\text{household expenditure}
=
\text{factor income}
+
\text{tariff revenue}
+
 B_s;
\]

and:

\[
M_s-X_s=B_s.
\]

For empirical trade data, FOB, CIF, tariffs, transport margins, and purchaser values must not be mixed without reconciliation.

---

# 12. Recommended next extensions

The safest sequence is:

1. replace artificial flows with a balanced three-region SAM;
2. preserve an explicit ROW and benchmark current accounts;
3. add intermediate inputs;
4. introduce a two-level Armington structure;
5. add government consumption, saving, and investment;
6. add trade and transport margins;
7. separate additional countries from ROW where policy incidence matters;
8. only then consider recursive dynamics or international capital mobility.

A useful rule is:

\[
\boxed{
\text{Do not add a new block until the smaller model replicates its benchmark.}
}
\]
