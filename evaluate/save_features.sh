srun -p long -w espresso-0-24 --gres=gpu:1 --pty \
python save_features.py --id baseline --model_weight ~/models/baseline.pth