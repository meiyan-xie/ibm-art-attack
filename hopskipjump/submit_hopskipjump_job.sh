#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/hopskipjump_slurm_mlp01_"${i}".sh
done
