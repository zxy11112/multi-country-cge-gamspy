# Deploying the GAMSPy CGE model on SCRP

This document describes how to move the model from the local Windows laptop to the SCRP Linux cluster and run it there.

## 1. What the model needs

| Resource | Requirement | Notes |
|---|---|---|
| CPU | 1–4 cores | PATH/MCP is essentially single-threaded; extra cores do not speed up one solve. |
| RAM | 4–8 GB | The 14 × 12 G20 model is tiny; 16 GB is comfortable. |
| Disk | < 1 GB | Source + data + outputs. |
| GPU | Not needed | The solver is purely CPU based. |
| Software | Python 3.11 + GAMSPy + GAMS + PATH solver license | GAMS must be installed separately. |

Recommended SCRP node for a single solve: **scrp-node-15, -16, or -17** (AMD Ryzen 7950X, 16 cores, 128 GB RAM, 1 TB NVMe). They are fast, lightly loaded for this workload, and avoid wasting a large EPYC node.

For **batch sensitivity sweeps** (many σ / tariff scenarios), use `scrp-node-4/5/23` (dual EPYC 9754, 256 cores, 1.5 TB RAM) and run many independent single-core jobs via SLURM job arrays.

## 2. Upload the project

Windows 默认没有 `rsync`，建议用 **zip + scp**。

### Option A: 一键脚本（推荐）

在 PowerShell 里进入 `D:\数据`：

```powershell
.\simple_multicountry_cge\deploy_to_scrp.ps1
```

脚本会：
1. 把 `simple_multicountry_cge/` 打成 zip（排除 `__pycache__`、`.pyc`、`outputs`）；
2. 用 `scp` 传到 SCRP 的 `~/cge_model/`；
3. `ssh` 登录并解压到 `~/cge_model/simple_multicountry_cge_new/`。

如果你的用户名不是 `xian kangwang`，先编辑脚本里的 `$User` 变量。

### Option B: 手动 zip + scp

```powershell
cd D:\数据

# 1. 打包（用 7-Zip 或 PowerShell）
Compress-Archive -Path "simple_multicountry_cge\*" -DestinationPath "cge_model_upload.zip"

# 2. 上传（用户名含空格，用引号包起来）
scp cge_model_upload.zip '"xian kangwang"@scrp-login-1.cuhk.edu.hk:~/cge_model/'

# 3. 登录解压
ssh '"xian kangwang"@scrp-login-1.cuhk.edu.hk' "mkdir -p ~/cge_model && cd ~/cge_model && unzip -o cge_model_upload.zip -d simple_multicountry_cge"
```

> Important: place the project in an **ASCII-only path** on the server, e.g. `~/cge_model/simple_multicountry_cge`. GAMS cannot handle non-ASCII or very long working directories (the same error you saw locally: `GetCurrentDir failed ... not ANSI`).

## 3. Install GAMS on SCRP

GAMSPy is only the Python API; the actual solver is GAMS. You need a GAMS distribution that includes the PATH MCP solver and a valid license.

1. Download the Linux x86_64 distribution from https://www.gams.com/download/ or use the CUHK license portal.
2. Extract to a fixed location, e.g. `~/gams/48.2`.
3. Add to `~/.bashrc`:

```bash
export PATH="$HOME/gams/48.2:$PATH"
```

4. Verify:

```bash
source ~/.bashrc
gams -Version
```

5. Place the GAMS license file (`gamslice.txt`) in `~/gams/48.2/` or set:

```bash
export GAMS_LICENSE="$HOME/gams/48.2/gamslice.txt"
```

## 4. Create the Python environment

On SCRP:

```bash
cd ~/cge_model/simple_multicountry_cge

# adjust the path to conda.sh if your miniconda is elsewhere
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate gamspy_env

# quick sanity check
python -c "import gamspy; print(gamspy.__version__)"
```

## 5. Run a single job interactively (test)

```bash
srun --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=00:30:00 --pty bash

cd ~/cge_model/simple_multicountry_cge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gamspy_env
export GAMSPY_WORKDIR="/tmp/gamspy_work_$USER"
mkdir -p "$GAMSPY_WORKDIR"
python run_on_server.py
exit
```

Expected wall time for the 14 × 12 G20 model: **~12–20 seconds** on a modern node.

## 6. Submit a batch job

```bash
cd ~/cge_model/simple_multicountry_cge
mkdir -p logs
sbatch scrp_job.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/gamspy_cge_*.out
```

## 7. Outputs

`run_on_server.py` writes a timestamped folder under `outputs/server_run_YYYYMMDD_HHMMSS/` containing:

- `welfare.csv` — welfare (% change) and equivalent variation (million USD).
- `chn_ele_exports.csv` — % change in CHN ELE exports by destination.
- `timing.json` — detailed timing breakdown.

The terminal log also prints the timing summary, e.g.:

```
=== Timing ===
  total:                 11.34s
  model build:           2.11s
  benchmark solve:       1.34s
  counterfactual solve:  7.75s
  homotopy steps:        8
  avg step:              0.95s
```

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GetCurrentDir failed ... not ANSI` | Working directory contains non-ASCII characters or is too long. | Use `~/cge_model/simple_multicountry_cge` and set `GAMSPY_WORKDIR=/tmp/gamspy_work_...` |
| `ValidationError: Error while reading the port!` | GAMS runtime failed to start, often due to path/encoding issues. | Same as above; also check GAMS license. |
| `gams: command not found` | GAMS is not on PATH. | Re-source `~/.bashrc` or set PATH in the job script. |
| PATH reports `Locally Infeasible` | Tolerance too tight for the data scale. | Increase `solver_tolerance` in `solve_model()` (default 1e-10; 1e-8 is safe). |
| Convergence gets slower / fails with larger shocks | Homotopy steps too few. | Increase `homotopy_steps` (e.g. 10–20) in `run_on_server.py`. |

## 9. Scaling up

- To run a sensitivity grid over σ, create a SLURM job array where each task reads a different `elasticities.csv`.
- To add more regions/sectors, just rebuild the data CSVs with `prepare_real_data_g20.py` and re-run; the model code does not need to change until you exceed PATH's memory limits (which is far beyond the current size).
