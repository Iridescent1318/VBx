#!/usr/bin/env bash

dataset=$1
ffactor=$2

DATA_DIR=''
VB_DIR=/home/liaozty20/VBx/VBx

if [[ $dataset = 'callhome97' ]]; then
    DATA_DIR=/home/liaozty20/callhome97
elif [[ $dataset = 'callhome2000' ]]; then
    DATA_DIR=/home/liaozty20/callhome2000
elif [[ $dataset = 'amicorpus' ]]; then
    DATA_DIR=/home/liaozty20/amicorpus
else
    echo "Wrong dataset. Only callhome97, callhome2000, amicorpus are supported."
    exit 1
fi

if [[ -d ${dataset}_sys/rttm_ffactor_${ffactor} ]]; then
    mkdir -p ${dataset}_sys/result
    SYS_RTTM_ALL=${dataset}_sys/result/${dataset}_sys_ffactor_${ffactor}_all.rttm
    REF_RTTM_ALL=${dataset}_sys/result/${dataset}_ref_all.rttm
    TMP_SYS_FILES=($(ls ${dataset}_sys/rttm_ffactor_${ffactor}))
    REF_PREFIX=${DATA_DIR}/rttm/
    SYS_PREFIX=${dataset}_sys/rttm_ffactor_${ffactor}/
    REF_FILES=${TMP_SYS_FILES[@]/#/$REF_PREFIX}
    SYS_FILES=${TMP_SYS_FILES[@]/#/$SYS_PREFIX}
    cat ${SYS_FILES} > ${SYS_RTTM_ALL}
    cat ${REF_FILES} > ${REF_RTTM_ALL}
    python ${VB_DIR}/../dscore/score.py --collar 0.25 --ignore_overlaps -r ${REF_RTTM_ALL} -s ${SYS_RTTM_ALL} > ${dataset}_sys/result/${dataset}_result_forgiving_${ffactor}
    python ${VB_DIR}/../dscore/score.py --collar 0.25 -r ${REF_RTTM_ALL} -s ${SYS_RTTM_ALL} > ${dataset}_sys/result/${dataset}_result_fair_${ffactor}
    python ${VB_DIR}/../dscore/score.py --collar 0.0 -r ${REF_RTTM_ALL} -s ${SYS_RTTM_ALL} > ${dataset}_sys/result/${dataset}_result_full_${ffactor}
else
    echo "${dataset}_sys/rttm_ffactor_${ffactor} not found!"
    exit 1
fi