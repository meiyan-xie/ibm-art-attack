#!/bin/bash -l
# NOTE the -l (login) flag!

# Set a job name
#SBATCH --job-name=lenet0
#SBATCH --output=lenet_100votes/lenet_0_{}.out
#SBATCH --error=lenet_100votes/lenet_0_{}.err
# Default in slurm

# Specify GPU queue
#SBATCH --partition=datasci
#SBATCH --mem=16G

# Request one gpu (max two)
##SBATCH --gres=gpu:1

#purge and load the correct modules
# module purge > /dev/null 2>&1
# module load cuda/10.2.89
# module load python/3.8

python3 -u lenet_cifar10_0.py
