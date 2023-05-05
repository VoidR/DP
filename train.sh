
dir_df=path/to
mkdir -p ${dir_df};
python fl_train.py --enc -k 5 --arch resnet20 --seed 0 --save-dir ${dir_df} | tee ${dir_df}/log.txt