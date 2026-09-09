#!/bin/bash
#SBATCH -p scrp
#SBATCH -w scrp-node-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /home/users/xiankangwang/gamspy_models/mc_retry_exports_%j.log

source /opt/network/anaconda3/etc/profile.d/conda.sh
conda activate gamspy_env
cd /home/users/xiankangwang/gamspy_models
for y in 2006 2007 2008 2009 2010 2011 2017 2018 2019; do
  echo "===== YEAR $y start $(date '+%H:%M:%S') ====="
  python mc_run.py --year $y --data-dir data_real_g20_$y \
    --n-scen 100000 --n-solve 10000 --workers 16 --mode mfn \
    --outdir outputs/mc_exports_2005_2019/mc_$y
done
echo ALL_RETRY_DONE
