import torch
from dataset.base_dataset import ObjBaseDataset
from dataset.dataloader import DataLoader
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from model.model_base import ModelBuilder, LearningModule
from model.parallel.replicate import patch_replication_callback
import os
import argparse
import h5py
import numpy as np
import math


def save_feature(args):
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor(arch=args.arch, weights=args.weight_init)
    fc_classifier = builder.build_classification_layer(args)
    network = LearningModule(feature_extractor, crit=[], cls=fc_classifier, output='feat')
    network.eval()

    dataset_train = ObjBaseDataset(args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    dataset_val = ObjBaseDataset(args.list_val, args, batch_per_gpu=args.batch_size_per_gpu)
    dataloader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    iter_train = iter(dataloader_train)
    iter_val = iter(dataloader_val)

    args.train_epoch_iters = \
        math.ceil(dataset_train.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    args.val_epoch_iters = \
        math.ceil(dataset_val.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    print('1 Train Epoch = {} iters'.format(args.train_epoch_iters))
    print('1 Val Epoch = {} iters'.format(args.val_epoch_iters))

    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    iterations = 0
    while iterations <= dataset_train.num_sample:
        batch_data = next(iter_train)
        if iterations % 1 == 0:
            print(iterations)
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())
        elif iterations == args.train_epoch_iters:
            features = features[:dataset_train.num_sample, :]
            labels = labels[:dataset_train.num_sample]
            break
        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
        iterations += 1

    f = h5py.File('data/test_feat/train_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.close()

    iterations = 0
    while iterations <= dataset_val.num_sample:
        batch_data = next(iter_val)
        if iterations % 1 == 0:
            print(iterations)
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())
        elif iterations == args.val_epoch_iters:
            features = features[:dataset_val.num_sample, :]
            labels = labels[:dataset_val.num_sample]
            break
        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
        iterations += 1

    f = h5py.File('data/test_feat/val_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.close()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--feat_dim', default=512)

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Novel/novel_obj_train_before_feat.odgt')
    parser.add_argument('--list_val',
                        default='./data/ADE/ADE_Novel/novel_obj_val_before_feat.odgt')
    parser.add_argument('--root_dataset',
                        default='../')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0, 1, 2, 3],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='input batch size')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--weight_init', default='')

    # Data related arguments
    parser.add_argument('--num_class', default=293)
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[200, 250],
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1500, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
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

    args = parser.parse_args()

    save_feature(args)
