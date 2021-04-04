#!/usr/bin/env bash

# for fea in $(ls sys/xvector)
# do
#     filename=$(echo "${fea}" | cut -f 1 -d '.')
#     touch ${filename}.txt
#     featbin/copy-feats ark:${filename}.ark ark,t:/${filename}.txt
# done

featbin/copy-feats ark:iacg.ark ark,t:- \
