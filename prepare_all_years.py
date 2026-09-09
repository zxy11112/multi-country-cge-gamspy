# -*- coding: utf-8 -*-
"""Generate the full benchmark data pipeline for years 2006-2019.
Runs per year: baseline tariffs -> BaTIS services -> main IO/trade data,
then copies elasticities.csv into each output directory.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\数据\数据")
PROJ = ROOT / "simple_multicountry_cge"
CLEAN = ROOT / "整理" / "02_清洗"

YEARS = range(2006, 2020)


def run(script: str, *args: str) -> None:
    print(f"\n########## {script} {' '.join(args)} ##########", flush=True)
    r = subprocess.run([sys.executable, script, *args],
                       cwd=PROJ, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
    print("\n".join(tail), flush=True)
    if r.returncode != 0:
        print(f"!!! {script} {args} FAILED rc={r.returncode}", flush=True)
        sys.exit(1)


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    for year in YEARS:
        print(f"\n==================== YEAR {year} ====================",
              flush=True)
        run("prepare_baseline_tariffs_g20.py", "--year", str(year))
        run("prepare_batis_services_g20.py", "--year", str(year))
        run("prepare_real_data_g20.py", "--year", str(year))
        out = PROJ / f"data_real_g20_{year}"
        shutil.copyfile(PROJ / "data_real_g20" / "elasticities.csv",
                        out / "elasticities.csv")
        n_files = len(list(out.glob("*.csv")))
        print(f"YEAR {year} DONE: {n_files} csv files in {out}", flush=True)
    print("\nALL YEARS DONE", flush=True)


if __name__ == "__main__":
    main()
