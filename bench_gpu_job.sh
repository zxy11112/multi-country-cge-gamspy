#!/bin/bash
#SBATCH -p scrp
#SBATCH -w scrp-node-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o /home/users/xiankangwang/gamspy_models/bench_gpu_%j.log

source /opt/network/anaconda3/etc/profile.d/conda.sh
conda activate gamspy_env
cd /home/users/xiankangwang/gamspy_models
python bench_forward.py
