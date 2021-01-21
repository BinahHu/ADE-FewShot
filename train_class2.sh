srun -p long -t 48:00:00 -w espresso-0-24 --gres=gpu:4 --pty \
python train.py --comment bbx_0.25class --ckpt ckpt/bbx_0.25class/ --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train_0.25class.json \
--list_val ./data/ADE/ADE_Base/base_img_val_0.25class.json \
--model_weight ~/models/baseline_0.25class.pth