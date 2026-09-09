#!/bin/bash
#SBATCH -p scrp
#SBATCH -w scrp-node-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 3:00:00
#SBATCH -o /home/users/xiankangwang/gamspy_models/train_nn_%j.log

source /opt/network/anaconda3/etc/profile.d/conda.sh
conda activate gamspy_env
cd /home/users/xiankangwang/gamspy_models
python train_welfare_nn.py
