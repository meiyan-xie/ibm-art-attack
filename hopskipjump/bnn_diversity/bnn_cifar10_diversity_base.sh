#!/bin/bash -l
# NOTE the -l (login) flag!

# Set a job name
#SBATCH --job-name=bnn_diversity
#SBATCH --output=bnn_diversity_out_1500/bnn_vote_{}.out
#SBATCH --error=bnn_diversity_out_1500/bnn_vote_{}.err
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

python3 -u bnn_diversity_script/bnn_cifar10_diversity_{}.py
