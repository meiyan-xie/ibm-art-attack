#!/bin/bash -l
# NOTE the -l (login) flag!

# Set a job name
#SBATCH --job-name=mlp01_100vote
#SBATCH --output=mlp01_100vote/mlp01_{}.out
#SBATCH --error=mlp01_100vote/mlp01_{}.err
# Default in slurm

# Specify GPU queue
#SBATCH --partition=datasci
#SBATCH --mem=8G

# Request one gpu (max two)
##SBATCH --gres=gpu:1

#purge and load the correct modules
# module purge > /dev/null 2>&1
# module load cuda/10.2.89
# module load python/3.8

python3 -u mlp01_cifar10.py
