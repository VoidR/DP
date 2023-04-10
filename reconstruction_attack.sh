mkdir -p checkpoints/dlg_enc_res20_5; mkdir -p checkpoints/dlg_enc_df_res20_5;

# 客户端未添加差分隐私
python fl_train.py --enc --dlg -b 1 --epochs 10 -k 5 --arch resnet20 --seed 0 --save-dir checkpoints/dlg_enc_res20_5 | tee checkpoints/dlg_enc_res20_5/log.txt

# 客户端添加差分隐私
python fl_train.py --enc --df --dlg -b 1 --epochs 10 -k 5 --arch resnet20 --seed 0 --save-dir checkpoints/dlg_enc_df_res20_5 | tee checkpoints/dlg_enc_df_res20_5/log.txt