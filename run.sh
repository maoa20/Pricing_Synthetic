#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 12
#SBATCH -o job.log
#SBATCH --error=job_error_CTGAN.txt

module load Python/3.8

source $HOME/sin_env/bin/activate

python -u $HOME/Pricing_Synthetic/train_CTGAN.py > log_CTGAN.txt