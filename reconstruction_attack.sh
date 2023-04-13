# mkdir -p checkpoints/dlg_enc_res20_5; mkdir -p checkpoints/dlg_enc_df_res20_5; 
mkdir -p checkpoints/dlg_enc_res56_5/figs; mkdir -p checkpoints/df_dlg_enc_res56_5/figs; 


# 客户端未添加差分隐私
python fl_train.py --enc --dlg -b 1 --epochs 1 -k 5 --arch resnet56 --seed 1234 --save-dir checkpoints/dlg_enc_res56_5 | tee checkpoints/dlg_enc_res56_5/log.txt

# 客户端添加差分隐私
echo "客户端添加差分隐私"
python fl_train.py --enc --df --dlg -b 1 --epochs 1 -k 5 --arch resnet56 --seed 1234 --save-dir checkpoints/df_dlg_enc_res56_5 | tee checkpoints/df_dlg_enc_res56_5/log.txt