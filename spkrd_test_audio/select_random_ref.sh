#!/bin/bash

dataset_name=$1
random_num=$2

if [[ $# -ne 2 ]]; then
    echo "Error: wrong number of params"
    exit
fi

mkdir -p ref/labs ref/rttm ref/wav
rm ref/labs/* ref/rttm/* ref/wav/*

fn_index=0
for fn in $(ls ~/${dataset_name}/wav); do
    file_names[$fn_index]=${fn%.*}
    fn_index=`expr $fn_index + 1`
done
cur_index=0
rand[0]=0

while [[ $cur_index -lt $random_num ]]; do
    rand_num=`expr $RANDOM % ${#file_names[*]}`
    while [[ $(for r in ${rand[*]}; do echo $r; done | grep $rand_num | wc -l) -ne 0 ]]; do
        rand_num=`expr $RANDOM % ${#file_names[*]}`
    done
    rand[$cur_index]=$rand_num
    rand_fn[$cur_index]=${file_names[${rand[$cur_index]}]}
    cur_index=`expr $cur_index + 1`
done

for rfn in ${rand_fn[*]}; do
    cp ~/${dataset_name}/wav/${rfn}.wav ref/wav
    cp ~/${dataset_name}/rttm/${rfn}.rttm ref/rttm
    cp ~/${dataset_name}/labs/${rfn}.lab ref/labs
done
