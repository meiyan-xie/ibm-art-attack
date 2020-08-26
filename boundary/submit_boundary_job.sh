#!/bin/bash
for i in {0..9}
do
    sbatch boundary_attack_bnn_"${i}".sh
done
