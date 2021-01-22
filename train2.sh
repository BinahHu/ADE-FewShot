srun -p short -t 48:00:00 -w espresso-0-18 --gres=gpu:4 --pty \
python train.py --comment rotation_pretrain_seg_attr_hie --ckpt ckpt/rotation_pretrain_seg_attr_hie --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train.json \
--model_weight ckpt/rotation_pretrain_seg_attr/net_epoch_7.pth