#!/usr/bin/env bash

dataset=amicorpus
ffactor=0.1
fvariable=xvector

nohup ./run_formal.sh ${dataset} ${ffactor} ${fvariable} > run_formal_${dataset}_${ffactor}_${fvariable}.log 2>&1 &