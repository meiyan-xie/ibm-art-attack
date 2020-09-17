#!/bin/bash
for i in {0..99}
do
    sed "s/{}/${i}/" bnn_cifar10_diversity_base.sh > bnn_100vote_slurm_script/bnn_cifar10_diversity_${i}.sh
done
