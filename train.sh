srun -p long --gres=gpu:4 --pty python train.py --comment seg_attr_hierarchy_10 --ckpt ckpt/seg_attr_hierarchy_10/ --start_epoch 1 --num_epoch 5 --model_weight ../models/seg_attr_10.pth
