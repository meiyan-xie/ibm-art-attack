#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/bnn_800_"${i}".sh
done
