srun -p long -w espresso-0-24 --gres=gpu:1 --pty \
python save_features.py --id baseline --model_weight ~/models/seg_hie_attr_MTL_Head.pth
