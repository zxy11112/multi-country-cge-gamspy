#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server-side wrapper for the GAMSPy CGE model.

Run this on the SCRP Linux cluster after the conda env and GAMS are installed:

    python run_on_server.py

It sets an ASCII-only GAMS working directory, runs the default G20 demo,
prints timing, and writes a timestamped report under outputs/.
"""
from __future__ import annotations

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# GAMS cannot tolerate non-ASCII / very long working directories. Force a
# short ASCII scratch dir (must exist). On Windows fall back to D:\gamspy_work.
_default_scratch = "/tmp/gamspy_work" if not sys.platform.startswith("win") else r"D:\gamspy_work"
GAMSPY_WORKDIR = os.environ.get("GAMSPY_WORKDIR", _default_scratch)
os.makedirs(GAMSPY_WORKDIR, exist_ok=True)
os.environ["GAMSPY_WORKDIR"] = GAMSPY_WORKDIR

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

import numpy as np
import pandas as pd

from multicountry_cge_gamspy import get_real_data_g20, solve_model


def main() -> int:
    t0 = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "outputs" / f"server_run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{timestamp}] GAMSPy CGE server run starting...")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {HERE}")
    print(f"GAMS scratch directory: {GAMSPY_WORKDIR}")

    # Load data and solve
    d = get_real_data_g20()
    regions = d["regions"]
    shock_sector = "ELE"
    shock_pp = 0.25
    o_i = regions.index("CHN")
    k_i = d["sectors"].index(shock_sector)
    s_i = regions.index("USA")
    target = float(d["tau0"][o_i, k_i, s_i]) + shock_pp
    shock = ("CHN", shock_sector, "USA", target)

    print(f"\nShock: USA tariff on CHN {shock_sector}: "
          f"{d['tau0'][o_i, k_i, s_i]:.2%} -> {target:.2%}")

    res = solve_model(
        data=d,
        numeraire=("CHN", "labor"),
        shock=shock,
        homotopy_steps=8,
    )

    b, c = res["benchmark"], res["counterfactual"]
    rr = c["real_income"] / b["real_income"]

    # Welfare table
    wdf = pd.DataFrame({
        "region": regions,
        "welfare_pct": (rr - 1) * 100,
        "EV_mUSD": d["I0"] * 1e4 * (rr - 1),
    })

    # Export changes
    export_rows = []
    for s2 in range(len(regions)):
        if s2 == o_i:
            continue
        chg = (c["x"][o_i, k_i, s2] / b["x"][o_i, k_i, s2] - 1) * 100
        export_rows.append({
            "destination": regions[s2],
            "pct_change": chg,
        })
    expdf = pd.DataFrame(export_rows)

    # Save outputs
    wdf.to_csv(out_dir / "welfare.csv", index=False)
    expdf.to_csv(out_dir / "chn_ele_exports.csv", index=False)
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(res["timing"], f, indent=2)

    # Print summary
    print("\n=== Results ===")
    print("\nwelfare and EV:")
    print(wdf.round(3).to_string(index=False))

    print(f"\nCHN {shock_sector} exports by destination, % change:")
    print(expdf.round(2).to_string(index=False))

    t = res["timing"]
    print("\n=== Timing ===")
    print(f"  total:                 {t['total_seconds']:.2f}s")
    print(f"  model build:           {t['build_seconds']:.2f}s")
    print(f"  benchmark solve:       {t['benchmark_solve_seconds']:.2f}s")
    print(f"  counterfactual solve:  {t['counterfactual_solve_seconds']:.2f}s")
    print(f"  homotopy steps:        {t['homotopy_steps']}")
    print(f"  avg step:              "
          f"{sum(t['per_step_seconds'])/max(1,len(t['per_step_seconds'])):.2f}s")
    print(f"  status benchmark:      {res['bench_status']}")
    print(f"  status counterfactual: {res['cf_status']}")

    elapsed = time.perf_counter() - t0
    print(f"\nWall time (incl. I/O): {elapsed:.2f}s")
    print(f"Outputs saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
