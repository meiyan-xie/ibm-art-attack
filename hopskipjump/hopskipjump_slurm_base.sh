#!/bin/bash -l
# NOTE the -l (login) flag!

# Set a job name
#SBATCH --job-name=resnet_5
#SBATCH --output=resnet_10votes/resnet_5_{}.out
#SBATCH --error=resnet_10votes/resnet_5_{}.err
# Default in slurm

# Specify GPU queue
#SBATCH --partition=datasci
##SBATCH --nodelist=node413
#SBATCH --mem=16G

# Request one gpu (max two)
#SBATCH --gres=gpu:1

#purge and load the correct modules
# module purge > /dev/null 2>&1
# module load cuda/10.2.89
# module load python/3.8

python3 -u resnet_cifar10_10vote_5.py
