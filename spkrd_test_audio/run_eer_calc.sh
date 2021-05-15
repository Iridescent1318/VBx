#!/usr/bin/env bash

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=/home/liaozty20/VBx/spkrd_test_audio/ref
VB_DIR=/home/liaozty20/VBx/VBx

mkdir -p sys/xvector sys/seg sys/rttm
rm thrs_*.txt

for audio in $(ls ${DIR}/wav)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo ${filename} > list.txt
    if !([ -f sys/xvector/${filename}.ark ]); then
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
    fi

    echo "EER calc Starts: "${filename}""
    # run variational bayes on top of x-vectors
    python ${VB_DIR}/eer_calc.py --file-name ${filename} \
        --in-rttm-dir ${DIR}/rttm \
        --rttm-file ${filename}.rttm \
        --xvec-ark-file sys/xvector/${filename}.ark \
        --segments-file sys/seg/${filename}.seg \
        --xvec-transform ${VB_DIR}/models/ResNet101_8kHz/transform.h5 \
        --plda-file ${VB_DIR}/models/ResNet101_8kHz/plda \
        --lda-dim 128 \
        --reg-seg-file sys/regseg/${filename}.regseg \
        --output-file thrs_592.txt
    echo "EER calc Ends: "${filename}""

done