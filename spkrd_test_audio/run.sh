#!/usr/bin/env bash

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=/home/liaozty20/VBx/spkrd_test_audio/ref
VB_DIR=/home/liaozty20/VBx/VBx

mkdir -p sys/xvector sys/seg sys/rttm sys/fbank

for audio in $(ls ${DIR}/wav)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo ${filename} > list.txt
    echo "X-vectors Extraction Starts: "${filename}""
    # run feature and x-vectors extraction
    python ${VB_DIR}/predict.py --in-file-list list.txt \
        --in-lab-dir ${DIR}/labs \
        --in-wav-dir ${DIR}/wav \
        --out-ark-fn sys/xvector/${filename}.ark \
        --out-seg-fn sys/seg/${filename}.seg \
        --backend pytorch \
        --weights ${VB_DIR}/models/ResNet101_8kHz/nnet/raw_195.pth \
        --model ResNet101 \
        --gpu $(${VB_DIR}/free_gpu.sh)
    echo "X-vectors Extraction Ends: "${filename}""

    echo "VB-HMM Starts: "${filename}""
    # run variational bayes on top of x-vectors
    python ${VB_DIR}/vbhmm.py --init AHC+VB \
        --out-rttm-dir sys/rttm \
        --xvec-ark-file sys/xvector/${filename}.ark \
        --segments-file sys/seg/${filename}.seg \
        --xvec-transform ${VB_DIR}/models/ResNet101_8kHz/transform.h5 \
        --plda-file ${VB_DIR}/models/ResNet101_8kHz/plda \
        --threshold -0.015 \
        --lda-dim 128 \
        --Fa 0.4 \
        --Fb 17 \
        --loopP 0.40
    echo "VB-HMM Ends: "${filename}""

    echo "Scoring Starts: "${filename}""
    # check if there is ground truth .rttm file
    REFDIR=ref/rttm/${filename}.rttm
    SYSDIR=sys/rttm/${filename}.rttm
    if [ -f $REFDIR ]
    then
        # run dscore
        # forgiving
        python ${VB_DIR}/../dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25 --ignore_overlaps
        # # fair
        # python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25
        # # full
        # python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.0
    fi
    echo "Scoring Ends: "${filename}""
done