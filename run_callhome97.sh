#!/usr/bin/env bash

export KALDI_ROOT='/home/liaozty20/kaldi'
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p callhome97_exp

for audio in $(ls /home/liaozty20/callhome97/wav)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > callhome97_exp/list.txt
      echo "X-vectors Extraction Starts: "${filename}""
      # run feature and x-vectors extraction
      python VBx/predict.py --in-file-list callhome97_exp/list.txt \
          --in-lab-dir /home/liaozty20/callhome97/labs \
          --in-wav-dir /home/liaozty20/callhome97/wav \
          --out-ark-fn callhome97_exp/${filename}.ark \
          --out-seg-fn callhome97_exp/${filename}.seg \
          --backend pytorch \
          --weights VBx/models/ResNet101_8kHz/nnet/raw_195.pth \
          --model ResNet101 \
          --gpu $(VBx/free_gpu.sh)
      echo "X-vectors Extraction Ends: "${filename}""

      echo "VB-HMM Starts: "${filename}""
      # run variational bayes on top of x-vectors
      python VBx/vbhmm.py --init AHC+VB \
          --out-rttm-dir callhome97_exp \
          --xvec-ark-file callhome97_exp/${filename}.ark \
          --segments-file callhome97_exp/${filename}.seg \
          --xvec-transform VBx/models/ResNet101_8kHz/transform.h5 \
          --plda-file VBx/models/ResNet101_8kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa 0.4 \
          --Fb 17 \
          --loopP 0.40
      echo "VB-HMM Ends: "${filename}""

      echo "Scoring Starts: "${filename}""
      # check if there is ground truth .rttm file
      REFDIR=/home/liaozty20/callhome97/rttm/${filename}.rttm
      SYSDIR=callhome97_exp/${filename}.rttm
      if [ -f $REFDIR ]
      then
          # run dscore
          # forgiving
          python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25 --ignore_overlaps
          # # fair
          # python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25
          # # full
          # python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.0
      fi
      echo "Scoring Ends: "${filename}""
done