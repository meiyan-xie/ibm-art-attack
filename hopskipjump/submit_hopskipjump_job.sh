#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/mlp_"${i}".sh
done
