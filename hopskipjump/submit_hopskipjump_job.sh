#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/bnn_random_0_"${i}".sh
done
