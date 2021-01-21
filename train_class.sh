srun -p short -t 48:00:00 -w espresso-0-22 --gres=gpu:4 --pty \
python train.py --comment bbx_attr_scene_0.75class --ckpt ckpt/bbx_attr_scene_0.75class/ --start_epoch 1 --num_epoch 8 \
--list_train ./data/ADE/ADE_Base/base_img_train_0.75class.json \
--list_val ./data/ADE/ADE_Base/base_img_val_0.75class.json \
--model_weight ~/models/bbx_attr_0.75class.pth