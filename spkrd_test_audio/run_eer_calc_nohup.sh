#!/usr/bin/env bash

DATASET=(callhome97 callhome2000 amicorpus)
DURATION=(15 30 60)

for dataset in ${DATASET[@]}
do
    for duration in ${DURATION[@]}
    do
        nohup ./run_eer_calc_all.sh ${dataset} 0.0 xvector ${duration} > logs/eer_${dataset}_${duration}.log 2>&1 &
    done
done
