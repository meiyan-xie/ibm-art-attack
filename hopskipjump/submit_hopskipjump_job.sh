#!/bin/bash
for i in {0..9}
do
    sbatch slurm_script/resnet_5_"${i}".sh
done
