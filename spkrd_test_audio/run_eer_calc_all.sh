#!/usr/bin/env bash

dataset=$1
ffactor=$2
fvariable=$3
duration=$4

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=/home/liaozty20/VBx/spkrd_test_audio/ref
VB_DIR=/home/liaozty20/VBx/VBx

DATA_DIR=''
WEIGHTS_DIR=${VB_DIR}/models/ResNet101_8kHz/nnet/raw_195.pth
XVEC_TRANS_DIR=${VB_DIR}/models/ResNet101_8kHz/transform.h5
PLDA_DIR=${VB_DIR}/models/ResNet101_8kHz/plda
REGSEG_END=${duration}
THRS_FILE=thrs/${dataset}_${REGSEG_END}

if [[ ${dataset} == "callhome97" ]]; then
    DATA_DIR=/home/liaozty20/callhome97
elif [[ ${dataset} == "callhome2000" ]]; then
    DATA_DIR=/home/liaozty20/callhome2000
elif [[ ${dataset} == "amicorpus" ]]; then
    DATA_DIR=/home/liaozty20/amicorpus
    WEIGHTS_DIR=${VB_DIR}/models/ResNet101_16kHz/nnet/raw_81.pth
    XVEC_TRANS_DIR=${VB_DIR}/models/ResNet101_16kHz/transform.h5
    PLDA_DIR=${VB_DIR}/models/ResNet101_16kHz/plda
else
    echo "Wrong dataset. Only callhome97, callhome2000, amicorpus are supported."
    exit 1
fi

echo "Running on ${dataset}, with factor ${ffactor} and variable ${fvariable}"
echo "data_dir: ${DATA_DIR}"
echo "weights_dir: ${WEIGHTS_DIR}"
echo "xvec_trans_dir: ${XVEC_TRANS_DIR}"
echo "plda_dir: ${PLDA_DIR}"
echo "regseg_end: ${REGSEG_END}"

mkdir -p ${dataset}_sys/xvector ${dataset}_sys/seg ${dataset}_sys/rttm_ffactor_${ffactor}_${fvariable} ${dataset}_sys/regseg_${REGSEG_END}

python3 select_reg_segs.py -i ${DATA_DIR}/rttm -o ${dataset}_sys/regseg_${REGSEG_END} -a regseg -e ${REGSEG_END} && echo "Selection of registered segments finished. Length: about ${REGSEG_END} seconds."

rm ${THRS_FILE}

for audio in $(ls ${DATA_DIR}/wav)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo ${filename} > list/${dataset}_sys_${ffactor}_${fvariable}_list.txt
    if !([ -f ${dataset}_sys/xvector/${filename}.ark ]); then
        echo "X-vectors Extraction Starts: ${filename}"
        # run feature and x-vectors extraction
        python ${VB_DIR}/predict.py --in-file-list list/${dataset}_sys_${ffactor}_${fvariable}_list.txt \
            --in-lab-dir ${DATA_DIR}/labs \
            --in-wav-dir ${DATA_DIR}/wav \
            --out-ark-fn ${dataset}_sys/xvector/${filename}.ark \
            --out-seg-fn ${dataset}_sys/seg/${filename}.seg \
            --backend pytorch \
            --weights ${WEIGHTS_DIR} \
            --model ResNet101 \
            --gpu $(${VB_DIR}/free_gpu.sh)
        echo "X-vectors Extraction Ends: ${filename}"
    fi

    echo "EER calc Starts: "${filename}""
    # run variational bayes on top of x-vectors
    REFDIR=${DATA_DIR}/rttm
    SYSDIR=${dataset}_sys/rttm_ffactor_${ffactor}_${fvariable}/${filename}.rttm
    python ${VB_DIR}/eer_calc.py --file-name ${filename} \
        --in-rttm-dir ${REFDIR} \
        --rttm-file ${filename}.rttm \
        --xvec-ark-file ${dataset}_sys/xvector/${filename}.ark \
        --segments-file ${dataset}_sys/seg/${filename}.seg \
        --xvec-transform ${XVEC_TRANS_DIR} \
        --plda-file ${PLDA_DIR} \
        --lda-dim 128 \
        --reg-seg-file ${dataset}_sys/regseg_${REGSEG_END}/${filename}.regseg \
        --output-file ${THRS_FILE}
    echo "EER calc Ends: "${filename}""
done