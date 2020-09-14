#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/scd01mlpbnn_"${i}".sh
done
