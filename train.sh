srun -p short -t 48:00:00 -w espresso-0-20 --gres=gpu:4 --pty \
python train.py --comment seg_attr_hie_drop --ckpt /home/zpang/ADE-FewShot/ckpt/seg_attr_hie_drop/ --start_epoch 1 --num_epoch 8 \
--list_train /home/zpang/ADE-FewShot/data/ADE/ADE_Base/base_img_train.json \