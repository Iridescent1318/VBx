#!/usr/bin/env bash

dataset=callhome2000
ffactor=0.05

nohup ./run_formal.sh ${dataset} ${ffactor} > run_formal_${dataset}_${ffactor}.log 2>&1 &