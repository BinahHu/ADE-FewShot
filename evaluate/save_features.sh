srun -p short -w espresso-0-16 --gres=gpu:1 --pty \
python save_features.py --id seg_attr_drop --model_weight ../ckpt/seg_attr_drop/net_epoch_7.pth