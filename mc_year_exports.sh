#!/bin/bash
#SBATCH -p scrp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /home/users/xiankangwang/gamspy_models/mc_year_exports_%j.log

YEAR=$1
source /opt/network/anaconda3/etc/profile.d/conda.sh
conda activate gamspy_env
cd /home/users/xiankangwang/gamspy_models
python mc_run.py --year $YEAR --data-dir data_real_g20_$YEAR \
  --n-scen 100000 --n-solve 10000 --workers 16 --mode mfn \
  --outdir outputs/mc_exports_2005_2019/mc_$YEAR
