srun -p long -w espresso-0-24 --gres=gpu:1 --pty \
python save_features.py --id baseline_halfclass --model_weight ~/models/baseline_halfclass.pth