import torch
from dataset.base_dataset import BaseDataset
from dataset.dataloader import DataLoader
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from model.builder import ModelBuilder
from model.base_model import BaseLearningModule
from model.parallel.replicate import patch_replication_callback
import os
import argparse
import h5py
import numpy as np
import math
from utils import selective_load_weights


def save_feature(args):
    # Network Builders
    builder = ModelBuilder(args)
    feature_extractor = builder.build_backbone()
    network = BaseLearningModule(args, feature_extractor, classifier=None)
    selective_load_weights(network, args.model_weight)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()
    network.eval()
    network.module.mode = 'feature'

    dataset_train = BaseDataset(args.data_train, args)
    dataset_val = BaseDataset(args.data_val, args)
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    loader_val = DataLoader(
        dataset_val, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    iter_train = iter(loader_train)
    iter_val = iter(loader_val)

    args.train_epoch_iters = \
        math.ceil(dataset_train.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    args.val_epoch_iters = \
        math.ceil(dataset_val.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    print('1 Train Epoch = {} iters'.format(args.train_epoch_iters))
    print('1 Val Epoch = {} iters'.format(args.val_epoch_iters))

    iterations = 0
    while iterations <= args.train_epoch_iters:
        batch_data = next(iter_train)
        if iterations % 10 == 0:
            print('{} / {}'.format(iterations, args.train_epoch_iters))
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())
        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
        iterations += 1

    f = h5py.File('data/test_feat/img_train_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.close()

    iterations = 0
    while iterations <= args.val_epoch_iters:
        batch_data = next(iter_val)
        if iterations % 10 == 0:
            print('{} / {}'.format(iterations, args.val_epoch_iters))
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())

        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())
            anchor = np.array(batch_data[0]['anchors'][0, :label.size, :])
            scale = np.tile(np.array(batch_data[0]['scales'][:label.size, :]), (label.size, 1))

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
        iterations += 1

    f = h5py.File('data/test_feat/img_val_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.close()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--architecture', default='resnet18')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--crop_height', default=1)
    parser.add_argument('--crop_width', default=1)

    # Path related arguments
    parser.add_argument('--data_train',
                        default='./data/ADE/ADE_Novel/novel_img_train.json')
    parser.add_argument('--data_val',
                        default='./data/ADE/ADE_Novel/novel_img_val.json')
    parser.add_argument('--root_dataset',
                        default='../')

    # optimization related argument
    parser.add_argument('--gpus', default=[0],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--model_weight', default='')

    # Data related arguments
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgShortSize', default=800, type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1500, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='max down sampling rate of the network')
    parser.add_argument('--down_sampling_rate', default=8, type=int,
                        help='down sampling rate of the segmentation label')
    parser.add_argument('--sample_type', default='inst',
                        help='instance level or category level sampling')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./checkpoint',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=10,
                        help='frequency to display')
    parser.add_argument('--log_dir', default="./log_base/",
                        help='dir to save train and val log')
    parser.add_argument('--comment', default="",
                        help='add comment to this train')
    parser.add_argument('--model', default="ckpt/crop_res_50/net_epoch_11.pth",
                        help='model to load')
    parser.add_argument('--max_anchor_per_img', default=100)

    args = parser.parse_args()

    save_feature(args)