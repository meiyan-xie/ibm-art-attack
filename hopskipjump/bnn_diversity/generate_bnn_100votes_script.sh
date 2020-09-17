#!/bin/bash
for i in {0..99}
do
    sed "s/{}/${i}/" bnn_cifar10_diversity_base.py > bnn_diversity_script/bnn_cifar10_diversity_${i}.py
done
