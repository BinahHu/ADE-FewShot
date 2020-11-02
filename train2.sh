srun -p short -t 48:00:00 -w espresso-0-22 --gres=gpu:4 --pty \
python train.py --comment seg_0.25_lr5e-3 --ckpt ckpt/seg_0.25_lr5e-3/ --start_epoch 1 --num_epoch 6 \
--list_train ./data/ADE/ADE_Base/base_img_train_0.25.json \
--model_weight ckpt/baseline_0.25/net_epoch_9.pth
