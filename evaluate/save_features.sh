srun -p short -w espresso-0-16 --gres=gpu:1 --pty \
python save_features.py --id rotation_pretrain_seg_attr --model_weight ../ckpt/rotation_pretrain_seg_attr/net_epoch_7.pth