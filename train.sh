srun -p short -t 48:00:00 -w espresso-0-20 --gres=gpu:4 --pty \
python train.py --comment seg_attr_hie_0.75 --ckpt ckpt/seg_attr_hie_0.75/ --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train_0.75.json \
--model_weight ~/models/seg_attr_0.75.pth
