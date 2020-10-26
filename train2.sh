srun -p short -t 48:00:00 -w espresso-0-22 --gres=gpu:4 --pty \
python train.py --comment seg_0.25class --ckpt ckpt/seg_0.25class/ --start_epoch 1 --num_epoch 10 \
--list_train ./data/ADE/ADE_Base/base_img_train_0.25class.json \
--list_val ./data/ADE/ADE_Base/base_img_val_0.25class.json \
--model_weight ckpt/baseline_0.25class/net_epoch_9.pth