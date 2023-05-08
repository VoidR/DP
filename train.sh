
dir=path/to56

python fl_train.py -k 5 --epochs 10 --arch resnet56 --seed 1234 --save-dir ${dir} | tee ${dir}/log.txt


CUDA_VISIBLE_DEVICES=3 python fl_train.py -k 5 --epochs 10 --arch resnet56 --seed 1234 --save-dir path/to56 | tee path/to56/log.txt