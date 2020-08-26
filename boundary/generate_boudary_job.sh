#!/bin/bash
for i in {0..9}
do
    sed "s/{}/${i}/" boundary_attack_bnn_base.sh > boundary_attack_bnn_${i}.sh
done
