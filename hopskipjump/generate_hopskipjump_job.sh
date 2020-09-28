#!/bin/bash
for i in {0..9}
do
    sed "s/{}/${i}/" hopskipjump_slurm_base.sh > slurm_script/bnn_random_0_${i}.sh
done
