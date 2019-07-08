import os
import time
import random
import argparse

import torch
import torch.nn as nn

from dataset.train_dataset import ObjTrainDataset, ImgTrainDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoaderIter, DataLoader
from utils import  AverageMeter, parse_devices

from model.model_base import ModelBuilder, LearningModule


def train(module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    module.train()

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        module.zero_grad()
        loss, acc = module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item() * 100)

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_feat: {:.6f}, lr_cls: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_feat, args.lr_cls,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

            # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        # adjust_learning_rate(optimizers, cur_iter, args)


def main(args):
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor()
    fc_classifier = builder.build_classification_layer(args)

    crit_cls = nn.NLLLoss(ignore_index=-1)
    crit_seg = nn.NLLLoss(ignore_index=-1)
    crit = [{'type': 'cls', 'crit': crit_cls, 'weight': 1},
            {'type': 'seg', 'crit': crit_seg, 'weight': 0}]

    dataset_train = ObjTrainDataset(args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers)
    )

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    iterator_train = iter(loader_train)
    optimizer_feat = torch.optim.SGD(feature_extractor.parameters(),
                                     lr=args.lr_feat, momentum=0.5)
    optimizer_cls = torch.optim.SGD(fc_classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5)
    optimizers = [optimizer_feat, optimizer_cls]
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}
    network = LearningModule(feature_extractor, crit, fc_classifier)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)

    for epoch in range(0, 4):
        train(network, iterator_train, optimizers, history, epoch, args)
    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/small_test/train_objs.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/small_test/')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=4, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=4, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_feat', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_cls', default=2e-2, type=float, help='LR')

    # Data related arguments
    parser.add_argument('--num_class', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[100, 150, 200, 300],
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument("--worst_ratio", default=100)
    parser.add_argument("--group_split", default=[1 / 2, 1, 2])

    args = parser.parse_args()

    main(args)