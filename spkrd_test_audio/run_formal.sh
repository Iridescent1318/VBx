#!/usr/bin/env bash

dataset=$1
ffactor=$2

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=/home/liaozty20/VBx/spkrd_test_audio/ref
VB_DIR=/home/liaozty20/VBx/VBx

DATA_DIR=''
WEIGHTS_DIR=${VB_DIR}/models/ResNet101_8kHz/nnet/raw_195.pth
XVEC_TRANS_DIR=${VB_DIR}/models/ResNet101_8kHz/transform.h5
PLDA_DIR=${VB_DIR}/models/ResNet101_8kHz/plda
REGSEG_END=15
VBHMM_FA=0.4
VBHMM_FB=17
VBHMM_LOOP=0.40

if [[ dataset -eq 'callhome97' ]]; then
    DATA_DIR=/home/liaozty20/callhome97
elif [[ dataset -eq 'callhome2000' ]]; then
    DATA_DIR=/home/liaozty20/callhome2000
elif [[ dataset -eq 'amicorpus' ]]; then
    DATA_DIR=/home/liaozty20/amicorpus
    WEIGHTS_DIR=${VB_DIR}/models/ResNet101_16kHz/nnet/raw_81.pth
    XVEC_TRANS_DIR=${VB_DIR}/models/ResNet101_16kHz/transform.h5
    PLDA_DIR=${VB_DIR}/models/ResNet101_16kHz/plda
    REGSEG_END=30
    VBHMM_FA=0.4
    VBHMM_FB=64
    VBHMM_LOOP=0.65
else
    echo 'Wrong dataset. Only callhome97, callhome2000, amicorpus are supported.'
    exit 1
fi

mkdir -p ${dataset}_sys/xvector ${dataset}_sys/seg ${dataset}_sys/rttm_ffactor_${ffactor} ${dataset}_sys/regseg

python3 select_reg_segs.py -i ${DATA_DIR}/rttm -o ${dataset}_sys/regseg -a regseg -e ${REGSEG_END} && echo "Selection of registered segments finished. Length: about ${REGSEG_END} seconds."

for audio in $(ls ${DATA_DIR}/wav)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo ${filename} > list.txt
    if !([ -f ${dataset}_sys/xvector/${filename}.ark ]); then
        echo "X-vectors Extraction Starts: "${filename}""
        # run feature and x-vectors extraction
        python ${VB_DIR}/predict.py --in-file-list list.txt \
            --in-lab-dir ${DATA_DIR}/labs \
            --in-wav-dir ${DATA_DIR}/wav \
            --out-ark-fn ${dataset}_sys/xvector/${filename}.ark \
            --out-seg-fn ${dataset}_sys/seg/${filename}.seg \
            --backend pytorch \
            --weights ${WEIGHTS_DIR} \
            --model ResNet101 \
            --gpu $(${VB_DIR}/free_gpu.sh)
        echo "X-vectors Extraction Ends: "${filename}""
    fi

    echo "VB-HMM Starts: "${filename}""
    # run variational bayes on top of x-vectors
    python ${VB_DIR}/vbhmm.py --file-name ${filename} \
        --init AHC+VB \
        --out-rttm-dir ${dataset}_sys/rttm_ffactor_${ffactor} \
        --xvec-ark-file ${dataset}_sys/xvector/${filename}.ark \
        --segments-file ${dataset}_sys/seg/${filename}.seg \
        --xvec-transform ${XVEC_TRANS_DIR} \
        --plda-file ${PLDA_DIR} \
        --threshold -0.015 \
        --lda-dim 128 \
        --Fa ${VBHMM_FA} \
        --Fb ${VBHMM_FB} \
        --loopP ${VBHMM_LOOP} \
        --fusion-factor ${ffactor} \
        --reg-seg-file ${dataset}_sys/regseg/${filename}.regseg
    echo "VB-HMM Ends: "${filename}""

    echo "Scoring Starts: "${filename}""
    # check if there is ground truth .rttm file
    REFDIR=${DATA_DIR}/rttm/${filename}.rttm
    SYSDIR=${dataset}_sys/rttm_ffactor_${ffactor}/${filename}.rttm
    if [ -f $REFDIR ]
    then
        # run dscore
        # forgiving
        # python ${VB_DIR}/../dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25 --ignore_overlaps
        # # fair
        # python ${VB_DIR}/../dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25
        # # full
        python ${VB_DIR}/../dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.0
    fi
    echo "Scoring Ends: "${filename}""
done