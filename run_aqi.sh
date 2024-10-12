#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result_aqi_D0.5.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 

# Run a Python script using Conda environment
python run.py --nsample 100 --dataset air_quality -waveblend_interpolation 1 --seed 1 --miu_ntc 0.1 --targetstrategy hybrid --missing_pattern block --src_adj_file AQI_b --tgt_adj_file AQI_t

