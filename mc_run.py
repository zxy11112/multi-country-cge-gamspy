# -*- coding: utf-8 -*-
"""
Monte Carlo tariff perturbation driver (one benchmark year per run).

For the year's baseline tariffs tau0 (decimals, [origin, sector, dest]):
    scenario n : tau_n = max(0, tau0 + eps_n),  eps_n ~ U[eps_lo, eps_hi]
                 drawn independently for every tariff element.

All n_scen counterfactual tariff matrices are generated and stored.
The first n_solve scenarios are solved with the G20 CGE model; outputs of
the remaining scenarios are stored as NaN (null).

Parallelism: --workers independent OS processes on ONE node. Each worker
builds its own model container, solves its slice sequentially with
warm-start (variable levels carry over between scenarios), and
checkpoint-saves its partial results every 200 scenarios.

Outputs in <outdir>:
    taus.npy        (n_scen, R, I, S) float32 -- all counterfactual tariffs
    results.npz     welfare_pct, ev_musd, gdp_real_ratio, gdp_nominal_ratio
                    (n_scen, R) float32 (NaN = unsolved); x_ratio
                    (n_scen, R, I, S) float32 -- bilateral flow ratio
                    counterfactual/benchmark (origin, sector, dest;
                    NaN = unsolved or zero baseline flow); status (n_scen,)
                    uint8 (0 ok / 1 non-optimal / 2 exception); solve_sec,
                    resid (n_scen,) float32
    summary.json    parameters, benchmark snapshot, worker stats
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCALE_INV = 1e4  # model units -> million USD
CKPT = 200       # checkpoint every K scenarios per worker


def _scratch(tag: str) -> str:
    base = os.environ.get("MC_SCRATCH_BASE") or (
        r"D:\gamspy_mc" if sys.platform.startswith("win") else "/tmp/gamspy_mc")
    p = os.path.join(base, f"work_{tag}_{os.getpid()}")
    os.makedirs(p, exist_ok=True)
    return p


def _worker(task):
    """Solve scenarios [start, end). Runs in its own spawned process."""
    wid, start, end, cfg = task
    os.environ["GAMSPY_WORKDIR"] = _scratch(f"w{wid}")
    import multicountry_cge_gamspy as mc  # fresh import in this process

    d = mc.get_real_data_g20(cfg["data_dir"], year=cfg["year"])
    regions, sectors = d["regions"], d["sectors"]
    R, I = len(regions), len(sectors)
    path_opts = {"convergence_tolerance": cfg["tol"]}

    h = mc.build_model(d, numeraire=(cfg["numeraire"], "labor"))
    cge, tau, V = h["cge"], h["tau"], h["vars"]
    cge.solve(solver="PATH", solver_options=path_opts)
    b = mc._extract(d, **V)
    bench = dict(real_income=b["real_income"], gdp_real=b["gdp_real"],
                 gdp_nominal=b["gdp_nominal"], I0=d["I0"], x0=b["x"],
                 gdp_real_musd=b["gdp_real"] * SCALE_INV,
                 gdp_nominal_musd=b["gdp_nominal"] * SCALE_INV)
    b_real, b_gdpr, b_gdpn = bench["real_income"], bench["gdp_real"], \
        bench["gdp_nominal"]
    b_x = bench["x0"]
    b_x_safe = np.where(b_x > 0, b_x, 1.0)

    # tau update frame; row order matches C-order [r, i, s] (as in _df3)
    rows = [(regions[a], sectors[c], regions[e])
            for a in range(R) for c in range(I) for e in range(R)]
    tdf = pd.DataFrame(rows, columns=["r", "i", "s"])
    tdf["value"] = 0.0

    taus = np.load(cfg["taus_path"], mmap_mode="r")
    nloc = end - start
    W = np.full((nloc, R), np.nan, np.float32)
    EV = np.full((nloc, R), np.nan, np.float32)
    GR = np.full((nloc, R), np.nan, np.float32)
    GN = np.full((nloc, R), np.nan, np.float32)
    XR = np.full((nloc, R, I, R), np.nan, np.float32)
    RESID = np.full(nloc, np.nan, np.float32)
    SEC = np.full(nloc, np.nan, np.float32)
    STAT = np.zeros(nloc, np.uint8)

    def save_partial():
        np.savez(cfg["partial_fmt"].format(wid), start=start, end=end,
                 bench=bench, W=W, EV=EV, GR=GR, GN=GN, XR=XR,
                 RESID=RESID, SEC=SEC, STAT=STAT)

    t_w0 = time.perf_counter()
    for k, n in enumerate(range(start, end)):
        tdf["value"] = np.asarray(taus[n]).reshape(-1)
        tau.setRecords(tdf)
        t0 = time.perf_counter()
        try:
            cge.solve(solver="PATH", solver_options=path_opts)
            SEC[k] = time.perf_counter() - t0
            c = mc._extract(d, **V)
            rr = c["real_income"] / b_real
            W[k] = (rr - 1) * 100
            EV[k] = d["I0"] * SCALE_INV * (rr - 1)
            GR[k] = c["gdp_real"] / b_gdpr
            GN[k] = c["gdp_nominal"] / b_gdpn
            XR[k] = np.where(b_x > 0, c["x"] / b_x_safe, np.nan)
            RESID[k] = np.abs(c["goods_log_resid"]).max()
            STAT[k] = 0 if "Optimal" in str(cge.status) else 1
        except Exception as e:  # noqa: BLE001
            STAT[k] = 2
            print(f"[w{wid}] scenario {n} exception: {e}", flush=True)
            try:  # reset to the benchmark point to restore warm start
                tdf["value"] = d["tau0"].reshape(-1)
                tau.setRecords(tdf)
                cge.solve(solver="PATH", solver_options=path_opts)
            except Exception:  # noqa: BLE001
                pass
        if (k + 1) % CKPT == 0:
            save_partial()
            print(f"[w{wid}] {k + 1}/{nloc} done", flush=True)
    save_partial()
    return dict(wid=wid, start=start, end=end, n=end - start,
                wall=time.perf_counter() - t_w0,
                nfail=int((STAT > 0).sum()),
                resid_max=float(np.nanmax(RESID)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2005)
    ap.add_argument("--data-dir", default=None,
                    help="default: data_real_g20_<year>")
    ap.add_argument("--n-scen", type=int, default=100000)
    ap.add_argument("--n-solve", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps-lo", type=float, default=-0.05)
    ap.add_argument("--eps-hi", type=float, default=0.1)
    ap.add_argument("--numeraire", default="CHN")
    ap.add_argument("--mode", choices=["mfn", "element"], default="mfn",
                    help="mfn: one eps per (destination,sector) tariff "
                         "(interpretation B); element: independent eps per "
                         "matrix element (interpretation A)")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--outdir", default=None,
                    help="default: outputs/mc_<year>")
    args = ap.parse_args()

    # resolve all paths BEFORE importing the model module (import chdirs)
    outdir = Path(args.outdir or f"outputs/mc_{args.year}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    taus_path = outdir / "taus.npy"
    partial_fmt = str(outdir / "partial_{}.npz")
    data_dir = args.data_dir or f"data_real_g20_{args.year}"

    os.environ["GAMSPY_WORKDIR"] = _scratch("main")
    import multicountry_cge_gamspy as mc  # noqa: PLC0415

    d = mc.get_real_data_g20(data_dir, year=args.year)
    tau0 = d["tau0"]
    R, I, S = tau0.shape
    regions = d["regions"]
    sectors = d["sectors"]
    print(f"[main] year={args.year} data_dir={data_dir} "
          f"tau0 shape=({R},{I},{S})", flush=True)

    # ---- 1. generate and store ALL counterfactual tariffs ----
    # mode "mfn" (default, interpretation B): one eps per (destination,
    # sector) tariff, applied to ALL origins alike (MFN preserved);
    # diagonal (domestic) and services tariffs stay at baseline (0).
    # mode "element" (interpretation A): independent eps per matrix element.
    t_gen0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    mm = np.lib.format.open_memmap(
        taus_path, mode="w+", dtype=np.float32,
        shape=(args.n_scen, R, I, S))
    CH = 2000
    o_ = np.arange(R)[:, None, None]
    i_ = np.arange(I)[None, :, None]
    s_ = np.arange(S)[None, None, :]
    srv = sectors.index("SRV")
    imp_mask = (o_ != s_) & (i_ != srv)          # (R, I, S) import cells
    for a in range(0, args.n_scen, CH):
        b_ = min(a + CH, args.n_scen)
        n = b_ - a
        if args.mode == "mfn":
            eps = rng.uniform(args.eps_lo, args.eps_hi, size=(n, I, S))
            block = np.repeat(tau0[None], n, axis=0)
            upd = np.maximum(0.0, block + eps[:, None, :, :])
            block[:, imp_mask] = upd[:, imp_mask]
        else:
            eps = rng.uniform(args.eps_lo, args.eps_hi, size=(n, R, I, S))
            block = np.maximum(0.0, tau0[None] + eps)
        mm[a:b_] = block
    mm.flush()
    del mm
    print(f"[main] generated {args.n_scen} scenarios ({args.mode}) -> "
          f"{taus_path} ({time.perf_counter() - t_gen0:.1f}s)", flush=True)

    # ---- 2. solve the first n_solve scenarios in parallel ----
    cfg = dict(data_dir=data_dir, taus_path=str(taus_path), tol=args.tol,
               numeraire=args.numeraire, partial_fmt=partial_fmt,
               year=args.year)
    bounds = np.linspace(0, args.n_solve, args.workers + 1).astype(int)
    tasks = [(w, int(bounds[w]), int(bounds[w + 1]), cfg)
             for w in range(args.workers) if bounds[w + 1] > bounds[w]]

    t_run0 = time.perf_counter()
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[main] worker {r['wid']} done: {r['n']} scen in "
                  f"{r['wall']:.0f}s, fails={r['nfail']}, "
                  f"resid_max={r['resid_max']:.1e}", flush=True)
    run_wall = time.perf_counter() - t_run0

    # ---- 3. merge partials into the final result arrays ----
    W = np.full((args.n_scen, R), np.nan, np.float32)
    EV = np.full((args.n_scen, R), np.nan, np.float32)
    GR = np.full((args.n_scen, R), np.nan, np.float32)
    GN = np.full((args.n_scen, R), np.nan, np.float32)
    XR = np.full((args.n_scen, R, I, S), np.nan, np.float32)
    RESID = np.full(args.n_scen, np.nan, np.float32)
    SEC = np.full(args.n_scen, np.nan, np.float32)
    STAT = np.zeros(args.n_scen, np.uint8)
    bench = None
    for w in range(args.workers):
        pf = Path(partial_fmt.format(w))
        if not pf.exists():
            continue
        p = np.load(pf, allow_pickle=True)
        s, e = int(p["start"]), int(p["end"])
        W[s:e] = p["W"]; EV[s:e] = p["EV"]; GR[s:e] = p["GR"]; GN[s:e] = p["GN"]
        XR[s:e] = p["XR"]
        RESID[s:e] = p["RESID"]; SEC[s:e] = p["SEC"]; STAT[s:e] = p["STAT"]
        if bench is None:
            bench = {k: p["bench"].item()[k] for k in
                     ("real_income", "gdp_real", "gdp_nominal", "I0", "x0",
                      "gdp_real_musd", "gdp_nominal_musd")}

    np.savez(outdir / "results.npz", regions=np.array(regions),
             welfare_pct=W, ev_musd=EV, gdp_real_ratio=GR,
             gdp_nominal_ratio=GN, x_ratio=XR, status=STAT, solve_sec=SEC,
             resid=RESID)

    solved_w = W[:args.n_solve]
    summary = dict(
        year=args.year, data_dir=data_dir, mode=args.mode,
        n_scen=args.n_scen,
        n_solve=args.n_solve, workers=args.workers, seed=args.seed,
        eps_lo=args.eps_lo, eps_hi=args.eps_hi, numeraire=args.numeraire,
        tolerance=args.tol, run_wall_seconds=run_wall,
        scenarios_per_second=round(args.n_solve / run_wall, 2),
        status_counts={str(k): int((STAT[:args.n_solve] == k).sum())
                       for k in (0, 1, 2)},
        resid_max=float(np.nanmax(RESID[:args.n_solve])),
        solve_sec_mean=float(np.nanmean(SEC[:args.n_solve])),
        solve_sec_p95=float(np.nanpercentile(SEC[:args.n_solve], 95)),
        bench={k: np.asarray(v).tolist() for k, v in (bench or {}).items()},
        welfare_pct_mean=np.nanmean(solved_w, axis=0).round(4).tolist(),
        welfare_pct_std=np.nanstd(solved_w, axis=0).round(4).tolist(),
        welfare_pct_min=np.nanmin(solved_w, axis=0).round(4).tolist(),
        welfare_pct_max=np.nanmax(solved_w, axis=0).round(4).tolist(),
        files=dict(taus=str(taus_path), results=str(outdir / "results.npz")),
    )
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[main] DONE in {run_wall:.0f}s "
          f"({summary['scenarios_per_second']}/s), "
          f"status={summary['status_counts']}, "
          f"resid_max={summary['resid_max']:.1e}", flush=True)
    print("[main] welfare_pct mean per region:",
          dict(zip(regions, summary["welfare_pct_mean"])), flush=True)
    print(f"[main] outputs: {outdir}", flush=True)


if __name__ == "__main__":
    main()
