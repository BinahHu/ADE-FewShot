"""
Training Code for the base feature extractor
Supporting several modes
"""
from dataset.base_dataset import ImgTrainDataset, ObjTrainDataset
from dataset.dataloader import DataLoader, DataLoaderIter
from dataset.collate import user_scattered_collate
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_train", default="./data/small_test/train_imgs.odgt")
    parser.add_argument("--root_dataset", default="./data/small_test/")
    parser.add_argument("--batch_size_per_gpu", default=2)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--imgSize", default=[100, 150, 200, 250], nargs="+", type=int)
    parser.add_argument("--imgMaxSize", default=1000, type=int)
    parser.add_argument("--worst_ratio", default=100)
    parser.add_argument('--padding_constant', default=8, type=int)
    parser.add_argument("--group_split", default=[1/2, 1, 2])
    parser.add_argument("--random_shuffle", default=True)
    parser.add_argument("--segm_downsampling_rate", default=8)
    parser.add_argument('--random_flip', default=True, type=bool)

    args = parser.parse_args()

    dataset_train = ImgTrainDataset(args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_train = DataLoader(
        dataset_train,
        batch_size=2,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        pin_memory=False)

    iterator_train = DataLoaderIter(loader_train)

    for i in range(4):
        batch_data = next(iterator_train)
        print(batch_data[0]['seg_label'].shape)
    print('hhh')

