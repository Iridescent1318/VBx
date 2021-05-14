#!/usr/bin/env bash

DATASET=(callhome97 callhome2000 amicorpus)
DURATION=(15 30 60)
PLDA_THRS=(-400 -450 -500 -550 -600 -650 -700 -750 -800)

for dataset in ${DATASET[@]}
do
    for duration in ${DURATION[@]}
    do
        for pldathrs in ${PLDA_THRS[@]}
        do
            nohup ./run_eer_calc_all.sh ${dataset} 0.0 xvector ${duration} ${pldathrs} > logs/eer_${dataset}_${duration}_${pldathrs}.log 2>&1 &
        done
    done
done
