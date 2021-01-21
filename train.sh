srun -p short -t 48:00:00 -w espresso-0-22 --gres=gpu:4 --pty \
python train.py --comment seg_attr_drop --ckpt ckpt/seg_attr_drop/ --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train_drop.json \
--model_weight ckpt/seg_drop/net_epoch_7.pth