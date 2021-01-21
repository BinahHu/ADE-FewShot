srun -p long -t 48:00:00 -w espresso-0-24 --gres=gpu:4 --pty \
python train.py --comment pure_rotation_pretrain --ckpt ckpt/pure_rotation_pretrain --start_epoch 0 --num_epoch 10 \
--list_train ./data/ADE/ADE_Base/base_img_train.json --lr_feat 1e-1 --lr_cls 0 \
--model_weight ''