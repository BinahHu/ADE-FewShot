srun -p long -w espresso-0-24 --gres=gpu:1 --pty \
python save_features.py --id seg_attr_hie_halfclass --model_weight ~/models/seg_attr_hie_halfclass.pth
