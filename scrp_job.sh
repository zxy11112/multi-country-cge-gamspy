#!/bin/bash
#SBATCH --job-name=gamspy_cge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gamspy_cge_%j.out
#SBATCH --error=logs/gamspy_cge_%j.err

# ----------------------------------------------------------------------
# SLURM job script for the GAMSPy multi-country CGE model on SCRP.
# ----------------------------------------------------------------------
# The model is PATH/MCP-based and mostly single-threaded; 4 cores and
# 16 GB RAM are plenty for the 14-region x 12-sector G20 version.
# For batch sensitivity sweeps, increase --array or run many small jobs
# instead of requesting a huge node.
# ----------------------------------------------------------------------

set -euo pipefail

# 1. locate project (adjust if you uploaded to a different path)
PROJECT_DIR="${PROJECT_DIR:-$HOME/cge_model/simple_multicountry_cge}"
cd "$PROJECT_DIR"
mkdir -p logs outputs

# 2. load conda/miniconda (SCRP uses miniconda3 by default)
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "/opt/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
{ echo "ERROR: cannot find conda.sh"; exit 1; }

conda activate gamspy_env

# 3. verify GAMS is available
if ! command -v gams &> /dev/null; then
    echo "ERROR: GAMS is not on PATH. Install GAMS and add it to PATH."
    exit 1
fi
echo "GAMS version: $(gams -Version | head -n 1)"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"

# 4. GAMS scratch directory: must be ASCII-only and writable
export GAMSPY_WORKDIR="/tmp/gamspy_work_${SLURM_JOB_ID:-$$}"
mkdir -p "$GAMSPY_WORKDIR"

# 5. run
python run_on_server.py

# 6. clean up scratch (optional; GAMS leaves small temp files)
rm -rf "$GAMSPY_WORKDIR"

echo "Job completed at $(date)"
