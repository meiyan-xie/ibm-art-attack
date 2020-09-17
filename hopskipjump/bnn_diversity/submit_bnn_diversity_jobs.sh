#!/bin/bash
for i in {0..99}
do
    sbatch bnn_100vote_slurm_script/bnn_cifar10_diversity_${i}.sh
done
