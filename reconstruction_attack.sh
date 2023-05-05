#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")

function runcode(){
    echo "Running code for ${1} with ${2} and ${3}"
    dir=checkpoints/${time}/${1}_${2}_${3}
    dir_df=checkpoints/${time}/${1}_${2}_${3}_df

    mkdir -p ${dir}/figs
    python fl_train.py --enc --dlg -b 1 --epochs 1 -k 5 --arch ${1} --opt ${2} --dist ${3} --iters 600 --dataset cifar10 --seed 1234 --save-dir ${dir} | tee ${dir}/log.txt
    
    mkdir -p ${dir_df}/figs
    python fl_train.py --enc --dlg --df -b 1 --epochs 1 -k 5 --arch ${1} --opt ${2} --dist ${3} --iters 600 --dataset cifar10 --seed 1234 --save-dir ${dir_df} | tee -a ${dir_df}/log.txt
}

nets=(resnet20 resnet32 resnet56)
opts=(LBFGS Adam)
dists=(norm cosine)
if [ "$#" -eq "1" ];then
    for opt in ${opts[@]}; do
        for dist in ${dists[@]}; do
            runcode ${nets[${1}]} $opt $dist
        done
    done
elif [ "$#" -eq "2" ];then
    echo "input not correct"
elif [ "$#" -eq "3" ];then
    runcode ${nets[${1}]} ${opts[${2}]} ${dists[${3}]}
else
    for net in ${nets[@]}; do
        for opt in ${opts[@]}; do
            for dist in ${dists[@]}; do
                runcode $net $opt $dist
            done
        done
    done
fi