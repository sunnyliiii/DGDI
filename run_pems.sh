#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result_aqi_D0.5.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 


# Run a Python script using Conda environment
python run.py --nsample 100 --dataset pems  -waveblend_interpolation 1 --seed 1 --miu_ntc 0.1  --src_adj_file pems08 --tgt_adj_file pems04


