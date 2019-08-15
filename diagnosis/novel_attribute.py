"""
Load the novel things and predict their attributes
Inspect if the attributes generalize to the novel categories
"""
import os
import time
import random
import argparse
import h5py
import json
import math
import copy

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../')

from dataset.base_dataset import BaseDataset
from dataset.novel_dataset import NovelDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoaderIter, DataLoader
from utils import AverageMeter, selective_load_weights

from model.builder import ModelBuilder
from model.base_model import BaseLearningModule
from model.parallel.replicate import patch_replication_callback
from model.novel_model import NovelClassifier


def instance_attr(args):
    builder = ModelBuilder(args)
    feature_extractor = builder.build_backbone()
    classifier = builder.build_classifier()
    # supervision
    if len(args.supervision) != 0:
        supervision_modules = []
        for supervision in args.supervision:
            supervision_modules.append({'name': supervision['name'],
                                        'module': getattr(builder, 'build_' + supervision['name'])()})
            supervision_modules[-1]['module'].mode = 'diagnosis'
        setattr(args, 'module', supervision_modules)
    network = BaseLearningModule(args, feature_extractor, classifier)
    selective_load_weights(network, args.model_weight)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()
    network.eval()
    network.module.mode = 'diagnosis'

    vargs = copy.deepcopy(args)
    delattr(vargs, 'supervision')
    dataset_base = BaseDataset(args.list_base, vargs)
    loader_base = DataLoader(
        dataset_base, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    iter_base = iter(loader_base)

    args.epoch_iters = \
        math.ceil(dataset_base.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    print('1 Base Epoch = {} iters'.format(args.epoch_iters))

    iterations = 0
    while iterations <= args.epoch_iters:
        batch_data = next(iter_base)
        if iterations % 10 == 0:
            print('{} / {}'.format(iterations, args.epoch_iters))
        if iterations == 0:
            preds, labels, imgs = network(batch_data)
            preds = np.array(preds.detach().cpu())
            labels = np.array(labels.cpu())
            imgs = np.array(imgs.cpu())
        else:
            pred, label, img = network(batch_data)
            pred = np.array(pred.detach().cpu())
            label = np.array(label.cpu())
            img = np.array(img.cpu())

            preds = np.vstack((preds, pred))
            labels = np.hstack((labels, label))
            imgs = np.hstack((imgs, img))
        iterations += 1

    f = h5py.File(args.output, 'w')
    f.create_dataset('preds', data=preds)
    f.create_dataset('labels', data=labels)
    f.create_dataset('imgs', data=imgs)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default='novel', help='base or novel')
    # Model related arguments
    parser.add_argument('--architecture', default='resnet18')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--crop_height', default=3)
    parser.add_argument('--crop_width', default=3)
    parser.add_argument('--model_weight', default='../ckpt/fine_tune_33/net_epoch_5.pth')
    parser.add_argument('--log', default='', help='load trained checkpoint')
    parser.add_argument('--num_base_class', default=189, type=int, help='number of classes')
    parser.add_argument('--num_novel_class', default=100, type=int)
    parser.add_argument('--padding_constant', default=8, type=int, help='max down sampling rate of the network')
    parser.add_argument('--down_sampling_rate', default=8, type=int, help='down sampling rate')
    parser.add_argument('--supervision', default='../supervision.json')

    # data loading arguments
    parser.add_argument('--list_base',
                        default='../data/ADE/ADE_Novel/novel_img_train.json')
    parser.add_argument('--list_novel',
                        default='../data/test_feat/img_val_feat.h5')
    parser.add_argument('--root_dataset', default='../../')
    parser.add_argument('--max_anchor_per_img', default=100)
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgShortSize', default=800, type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1500, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--base_epoch_iters', default=10)
    parser.add_argument('--novel_epoch_iters', default=10)

    # running arguments
    parser.add_argument('--gpus', default=[0], help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='input batch size')
    parser.add_argument('--epoch_iters', default=20, type=int)

    # logging
    parser.add_argument('--display_iter', type=int, default=10, help='frequency to display')
    parser.add_argument('--output', type=str, default='data/novel_attr.h5')

    args = parser.parse_args()
    args.supervision = json.load(open(args.supervision, 'r'))
    instance_attr(args)