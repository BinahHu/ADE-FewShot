srun -p short -t 48:00:00 -w espresso-0-22 --gres=gpu:4 --pty \
python train.py --comment seg_hie_attr_MTL_Head --ckpt ckpt/seg_hie_attr_MTL_Head/ --start_epoch 1 --num_epoch 15 \
--lr_all 0.1 --model_weight ~/models/baseline_ATT_Head2.pth
