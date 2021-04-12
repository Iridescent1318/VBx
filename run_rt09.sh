#!/usr/bin/env bash

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p rt09_exp

mkdir -p result

for audio in $(ls /home/liaozty20/rt09/wav/16k)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > rt09_exp/list.txt
      echo "X-vectors Extraction Starts: "${filename}""
      # run feature and x-vectors extraction
      python VBx/predict.py --in-file-list rt09_exp/list.txt \
          --in-lab-dir /home/liaozty20/rt09/labs \
          --in-wav-dir /home/liaozty20/rt09/wav/16k \
          --out-ark-fn rt09_exp/${filename}.ark \
          --out-seg-fn rt09_exp/${filename}.seg \
          --backend pytorch \
          --weights VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
          --model ResNet101 \
          --gpus $(VBx/free_gpu.sh)
      echo "X-vectors Extraction Ends: "${filename}""

      echo "VB-HMM Starts: "${filename}""
      # run variational bayes on top of x-vectors
      python VBx/vbhmm.py --init AHC+VB \
          --out-rttm-dir rt09_exp \
          --xvec-ark-file rt09_exp/${filename}.ark \
          --segments-file rt09_exp/${filename}.seg \
          --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
          --plda-file VBx/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa 0.3 \
          --Fb 17 \
          --loopP 0.99
      echo "VB-HMM Ends: "${filename}""

      echo "Scoring Starts: "${filename}""
      # check if there is ground truth .rttm file
      REFDIR=/home/liaozty20/rt09/rttm/${filename}.rttm
      SYSDIR=rt09_exp/${filename}.rttm
    #   if [ -f /home/liaozty20/AMI\-diarization\-setup/only_words/rttms/test/${filename}.rttm ]
    #   then
    #       # run dscore
    #       # forgiving
    #       python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25 --ignore_overlaps
    #       # fair
    #       python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25
    #       # full
    #       python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.0
    #   fi
      echo "Scoring Ends: "${filename}""
done