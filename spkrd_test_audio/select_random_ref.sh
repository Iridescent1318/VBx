#!/bin/bash

dataset_name=$1
random_num=$2

if [[ $# -ne 2 ]]; then
    echo "Error: wrong number of params"
    exit
fi

rm -r ref sys
mkdir -p ref/labs ref/rttm ref/wav
mkdir sys

fn_index=0
for fn in $(ls ~/${dataset_name}/wav); do
    file_names[$fn_index]=${fn%.*}
    fn_index=`expr $fn_index + 1`
done
cur_index=0
rand[0]=0

if [[ $fn_index -le $random_num ]]; then
    echo "Warning: random num is greater than num of all wavs. Setting random_num=file_num"
    random_num=$fn_index
fi

rand_fn=$(python3 gen_rand_num.py -n ${fn_index} -c ${random_num})

for rfn in ${rand_fn[*]}; do
    cp ~/${dataset_name}/wav/${file_names[rfn]}.wav ref/wav
    cp ~/${dataset_name}/rttm/${file_names[rfn]}.rttm ref/rttm
    cp ~/${dataset_name}/labs/${file_names[rfn]}.lab ref/labs
done

python3 select_reg_segs.py -i ref/rttm -o sys/regseg -a regseg
