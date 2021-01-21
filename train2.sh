srun -p long -t 48:00:00 -w espresso-0-24 --gres=gpu:4 --pty \
python train.py --comment rotation_pretrain_baseline --ckpt ckpt/rotation_pretrain_baseline --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train.json --lr_feat 1e-1 --lr_cls 1e-1 \
--supervision null.json \
--model_weight ckpt/pure_rotation_pretrain/net_epoch_9.pth