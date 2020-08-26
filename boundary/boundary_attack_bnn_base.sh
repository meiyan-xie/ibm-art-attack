#!/bin/bash -l
# NOTE the -l (login) flag!

# Set a job name
#SBATCH --job-name=bnn
#SBATCH --output=bnn/boundary_bnn_{}.out
#SBATCH --error=bnn/boundary_bnn_{}.err
# Default in slurm

# Specify GPU queue
#SBATCH --partition=datasci
#SBATCH --mem=8G

# Request one gpu (max two)
## SBATCH --gres=gpu:1

#purge and load the correct modules
module purge > /dev/null 2>&1
# module load cuda
# module load python/3.8

python3 -u boundary_attack_bnn.py
