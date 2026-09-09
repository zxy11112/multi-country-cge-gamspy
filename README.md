# Multi-Country CGE Model with GAMSPy (PATH)

A static multi-region, multi-sector computable general equilibrium (CGE) model
built with **GAMSPy** and solved with **PATH** (MCP). The model is calibrated
to **real 2022 data** for **14 regions x 12 sectors**, with intermediate
inputs, a nested Armington trade structure, ad valorem tariffs, and an
explicit Rest-of-World region.

The default experiment replicates a tariff escalation scenario:
the United States raises its tariff on Chinese electronics (ELE) by
25 percentage points (4.09% -> 29.09%), and the model traces the resulting
welfare, trade-diversion, and price effects across all 14 regions.

---

## 1. Model structure

| Dimension | Content |
|---|---|
| Regions (14) | CHN, USA, EU27, JPN, KOR, IND, CAN, MEX, BRA, ZAF, RUS, SAU, AUS, ROW |
| Sectors (12) | AGF, MIN, ENR, CHM, TXL, WDP, NMM, MAC, ELE, VEH, OTM, SRV |
| Factors (2) | labor, capital (mobile across sectors, immobile across regions) |

Regions are aggregated from OECD economies; the European Union enters as a
single region (intra-EU trade is excluded via extra-EU control totals), and
all remaining economies form the endogenous ROW region.

### Production

Sectoral output uses Cobb-Douglas value added over labor and capital, combined
with intermediate inputs through fixed (Leontief) input coefficients taken
from the benchmark input-output table:

```
Y[r,i] = CD_va(F[labor], F[capital]) ,  intermediates a[j,i] * Y[r,i]
```

Unit cost equals producer price under perfect competition (zero-profit
condition).

### Trade: nested Armington

Each destination-sector composite is a two-level CES aggregate:

1. **Bottom nest** (elasticity `sigma[i]`): substitution across foreign
   origins.
2. **Top nest** (elasticity `sigma_d[i]`): substitution between the domestic
   variety and the import aggregate.

Sectoral elasticities are literature-based estimates
(`data_real_g20/elasticities.csv`, see `elasticities_notes.md` for sources);
they can be edited for sensitivity analysis.

### Tariffs and income

The purchaser price in destination `s` is `q[r,i,s] = (1 + tau[r,i,s]) * p[r,i]`.
Ad valorem tariff revenue is rebated lump-sum to the destination household.
Domestic transactions carry no tariff (`tau[s,i,s] = 0`).

Regional household income:

```
I[s] = sum_f w[s,f] * Fbar[s,f]  +  T[s]  +  B[s]
```

where `T[s]` is tariff revenue and `B[s]` is a fixed net foreign transfer
(calibrated to the benchmark current-account imbalance, summing to zero
globally). Denominating `B` in a fixed transfer-unit price keeps the system
homogeneous of degree zero, so the numeraire choice does not affect real
allocations.

### Equilibrium (MCP formulation)

Unknowns are producer prices, factor prices, output levels, composite prices,
consumption, bilateral demand, and income. The system contains:

- zero-profit (unit cost >= producer price, per unit activity);
- goods-market clearing for every origin-sector variety;
- factor-market clearing in every region;
- household income balance;
- Armington price/quantity definitions.

One goods-market equation is omitted (redundant by Walras' law after the
numeraire `w[CHN,labor] = 1` is fixed) and checked ex post. All variables
enter in **logarithms**, guaranteeing positivity.

---

## 2. Data

The benchmark is built from public international datasets
(benchmark year 2022, million USD):

| Input | Source | Used for |
|---|---|---|
| Inter-country input-output tables | OECD ICIO / IOTs (DOMIMP, value added) | output, intermediates, final demand, factor payments |
| Bilateral goods trade | UN Comtrade, SITC (reported + expansion reporters; mirror flows where a country does not report) | destination shares of goods exports |
| Bilateral services trade | WTO-OECD BaTIS (balanced values) | services bilateral flows |
| Tariffs | World Bank WITS / TRAINS, HS6 (MFN simple averages per sector; AHS indicators as fallback) | baseline tariff matrix |
| Effective-tariff calibration | IMF GFS customs revenue (G1151) | scaling factors anchoring tariffs to observed revenue |

Calibration reconciles, for every region: output = factor payments +
intermediate costs; household expenditure = factor income + tariff revenue +
net transfers; and imports - exports = B[s]. CSV inputs are in
`data_real_g20/` and are fully reproducible from the raw sources with the
scripts in Section 4.

---

## 3. Installation and execution

Requirements: Python 3.11+, a GAMS installation with a valid license, and
GAMSPy. See `environment.yml`.

```bash
pip install -r requirements.txt        # numpy, pandas, ...
pip install gamspy                     # plus your GAMS license

# run benchmark verification + the US 25pp tariff on CHN ELE experiment
python run_g20_demo.py
```

The demo performs three checks:

1. **Benchmark replication** -- solving at baseline tariffs reproduces the
   calibrated data (weighted deviation of bilateral flows ~2e-6; max price
   deviation ~3e-5, consistent with CES calibration from rounded shares).
2. **Goods-market clearing** under the shock (max log residual 8.3e-13).
3. **Numeraire invariance** -- the same experiment solved with
   `w[CHN,labor] = 1` and `w[USA,labor] = 1` gives identical real
   allocations (max deviation 3.2e-5).

A pre-generated log is in `run_g20_demo_log.txt`; a results bundle
(welfare, timing, export changes) is in
`outputs/server_run_20260902_154931/`.

---

## 4. Reproducing the benchmark data (optional)

`data_real_g20/` already contains the calibrated 2022 dataset, so running
the model does not require re-downloading anything. To rebuild it from raw
sources:

```bash
python download_comtrade_g20.py        # UN Comtrade goods (needs API key)
python download_comtrade_expansion.py  #   additional reporters
python download_g1151_row.py           # IMF GFS customs revenue
python download_wits_expansion.py      # WITS HS6 tariffs
python download_wits_expansion_retry.py  #   resume truncated downloads

python clean_comtrade_g20.py           # tidy Comtrade long table
python clean_comtrade_expansion.py

python compute_row_tariff_factor.py    # effective-tariff factors (IMF G1151)
python prepare_baseline_tariffs_g20.py # sector-level baseline tariffs
python prepare_batis_services_g20.py   # BaTIS bilateral services flows
python prepare_effective_tariffs_g20.py
python prepare_real_data_g20.py        # main calibration -> data_real_g20/
```

Note: the WITS API is slow and rate-limited; the retry script re-downloads
truncated files chapter by chapter.

---

## 5. Illustrative result: US +25pp tariff on Chinese electronics

Baseline US tariff on CHN electronics is 4.09%; the experiment raises it to
29.09%. All other tariffs stay at baseline. Key outcomes:

**Welfare (real income change) and equivalent variation (million USD):**

| Region | Welfare % | EV (mUSD) |
|---|---|---|
| CHN | -0.073 | -10,130 |
| USA | +0.041 | +9,109 |
| EU27 | +0.151 | +17,904 |
| JPN | +0.122 | +4,051 |
| KOR | +0.140 | +1,988 |
| IND | +0.027 | +800 |
| CAN | +0.117 | +1,883 |
| MEX | +0.014 | +189 |
| BRA | +0.121 | +1,691 |
| ZAF | +0.138 | +411 |
| RUS | +0.258 | +3,698 |
| SAU | +0.374 | +2,444 |
| AUS | +0.214 | +2,466 |
| ROW | +0.192 | +23,484 |

**Trade diversion**: CHN electronics exports to the USA fall by 59.2%,
while every third market absorbs more (EU27 +2.4%, JPN +2.0%, KOR +2.0%,
ROW +2.0%, ...). The tariff reallocates market share rather than eliminating
Chinese electronics exports.

These numbers are model outputs of a specific 2022 calibration, not
reduced-form estimates; the magnitudes scale with the assumed Armington
elasticities.

---

## 6. Monte Carlo counterfactuals, 2005-2019

`mc_run.py` runs a Monte Carlo tariff-perturbation experiment on the SCRP
cluster, once per benchmark year 2005-2019:

1. Generate and store 100,000 counterfactual tariff matrices
   `tau_n = max(0, tau0 + eps_n)`, with `eps_n ~ U[-0.05, +0.10]` drawn
   independently per (destination, sector) tariff and applied to all origins
   alike (MFN preserved); domestic and services tariffs stay at baseline.
2. Solve the first 10,000 scenarios with the CGE model (16 parallel worker
   processes per year, warm-started); the remaining 90,000 scenarios are
   stored with null outputs.

Per scenario the solver stores 14-region `welfare_pct`, `ev_musd`,
real/nominal GDP ratios, and the full bilateral trade-flow ratio tensor
`x_ratio` (origin x sector x destination). All 15 years x 10,000 scenarios
solved with zero failures (max goods-market residual 3.4e-10).

```
mc_run.py               Monte Carlo driver (generation + parallel solve)
mc_year_exports.sh      Slurm script, one year per job
mc_retry_exports.sh     Slurm script, sequential fallback (license limits)
prepare_all_years.py    build the 2005-2019 benchmark datasets
download_wits_yearly.py WITS tariff download per year
mc_results/             MANIFEST.md + manifest_summary.csv + per-year
                        summary.json (parameters, benchmark snapshot, stats)
```

The full per-year tensors (`results.npz` ~920 MB, `taus.npy` ~900 MB per
year, 27 GB total) live on the cluster, not in this repository; the
`summary.json` files in `mc_results/` contain the benchmark snapshots and
distributional statistics needed to reproduce the headline numbers.

---

## 7. Neural surrogate for welfare response

`train_welfare_nn.py` fits a fully connected network to the 150,000 solved
Monte Carlo scenarios, mapping

```
(168 MFN tariffs [14 destinations x 12 sectors] + 15 year one-hots)
    -> 14-region welfare change (%)
```

Architecture: `183 -> 512 -> 512 -> 256 -> 128 -> 14`, ReLU, Adam
(lr 1e-3, batch 512), early stopping at epoch 303. Test-set performance
(15,000 held-out scenarios): mean R2 = 0.9996, overall RMSE = 0.0076
percentage points, verified with an independent split.

Inference replaces one PATH solve (~410 ms) with one forward pass:
~170 us single-scenario (CPU), ~0.17 us per scenario in a 100k batch on a
single RTX 3060. The surrogate is an interpolation tool: it is valid only
inside the training domain (tariff perturbations within [-5%, +10%] of the
2005-2019 benchmarks). Out-of-domain scenarios should be solved with the
CGE model directly.

```
train_welfare_nn.py     training (dataset cache, standardization, metrics)
bench_forward.py        inference-speed benchmark (CPU/GPU)
infer_demo.py           prediction vs ground-truth demo
nn_job.sh / nn_job_gpu.sh / bench_gpu_job.sh   Slurm scripts
welfare_nn/             trained weights (.pt, .npz) + metrics.json
```

---

## 8. Repository layout

```
multicountry_cge_gamspy.py   model: calibration, MCP construction, PATH solve
                             (2026-09 update: multi-year benchmarks, GDP
                             outputs, MC/NN support functions)
run_g20_demo.py              verification + US/CHN ELE tariff experiment
run_g20_demo_log.txt         pre-generated run log
data_real_g20/               calibrated 2022 benchmark (CSV inputs)
prepare_*_g20.py             benchmark data pipeline (Section 4)
download_*.py / clean_*.py   raw data acquisition and tidying
compute_row_tariff_factor.py effective-tariff calibration (IMF GFS)
outputs/server_run_*/        pre-generated results bundle
mc_run.py / mc_results/      Monte Carlo experiment, 2005-2019 (Section 6)
train_welfare_nn.py / welfare_nn/   neural surrogate (Section 7)
scrp_job.sh                  example Slurm batch script
DEPLOY.md                    cluster deployment notes
environment.yml              conda environment
```

---

## 9. Limitations

- Static model: no capital accumulation, dynamics, or supply-side investment
  response.
- Armington elasticities are literature-based; welfare magnitudes are
  sensitive to them.
- ROW is a single synthetic region; incidence inside ROW is not identified.
- Tariffs are the only policy instrument (no NTMs, subsidies, or exchange
  rates).
- Benchmark calibrations cover 2005-2019 (Section 6); the headline demo is
  the 2022 calibration. The model does not track growth between benchmark
  and policy date.

Natural extensions: additional benchmark years, endogenous labor supply,
government saving-investment closure, and disaggregating ROW into
individual countries where policy incidence matters.

---

## Citation

If you use this code, please cite the repository and acknowledge the data
sources listed in Section 2 (OECD, UN Comtrade, WTO-OECD BaTIS, World Bank
WITS, IMF GFS).
