#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/resnet_1991_"${i}".sh
done
