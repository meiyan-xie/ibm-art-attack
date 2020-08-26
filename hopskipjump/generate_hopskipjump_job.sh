#!/bin/bash
for i in {0..9}
do
    sed "s/{}/${i}/" hopskipjump_slurm_base.sh > slurm_script/hopskipjump_slurm_mlp01_${i}.sh
done
