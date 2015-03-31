#!/bin/bash

for saturate in 300 600 900 1200; do
    for decay in 0.001 0.003 0.01 0.03; do
        for min_delta in 0.001 0.01; do
            for max_delta in 0.1 1 10; do
                sed "s/{saturate}/$saturate/g" mnist.yaml |
                sed "s/{decay}/$decay/g" |
                sed "s/{min_delta}/$min_delta/g" |
                sed "s/{max_delta}/$max_delta/g" > /tmp/mnist.yaml
                ../../train.py /tmp/mnist.yaml
                ../../plot_monitor.py --out /var/www/dumped/mnist-rprop-$min_delta-$max_delta-$saturate-$decay.jpg --channel DC /tmp/mnist-$min_delta-$max_delta-$saturate-$decay.pkl
            done
        done
    done
done
