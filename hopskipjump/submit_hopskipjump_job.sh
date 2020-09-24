#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/lenet_0_"${i}".sh
done
