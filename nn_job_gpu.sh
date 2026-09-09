#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=c32g8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o /home/users/xiankangwang/gamspy_models/train_nn_gpu_%j.log

source /opt/network/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /home/users/xiankangwang/gamspy_models
python train_welfare_nn.py
