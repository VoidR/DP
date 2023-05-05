#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")

function runcode(){
    echo "Running code for ${1} and index=${2}"
    dir=checkpoints/${time}/${1}_${2}

    mkdir -p ${dir}/figs
    python test.py  --arch ${1}  --index ${2} --save-dir ${dir} | tee ${dir}/log.txt
}

nets=(resnet20 resnet32 resnet56)


for idx in {1..1000};do 
    # echo $idx
    runcode resnet20 $idx
done
