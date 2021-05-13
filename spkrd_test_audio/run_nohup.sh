#!/usr/bin/env bash

DATASET=(callhome97 callhome2000 amicorpus)
FFACTOR_X=(0.0 0.05 0.1 0.15 0.02)
FFACTOR_S=(0.002 0.004 0.006 0.008 0.01)

for dataset in ${DATASET[@]}
do
    for ffactor in ${FFACTOR_X[@]}
    do
        nohup ./run_formal.sh ${dataset} ${ffactor} xvector > logs/run_formal_${dataset}_${ffactor}_xvector.log 2>&1 &
    done
    for ffactor in ${FFACTOR_S[@]}
    do
        nohup ./run_formal.sh ${dataset} ${ffactor} second_stat_iter > logs/run_formal_${dataset}_${ffactor}_second_stat_iter.log 2>&1 &
    done
done
