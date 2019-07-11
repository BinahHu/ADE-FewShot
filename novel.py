import os
import time
import random
import argparse
import json
import math

import torch
import torch.nn as nn
from dataset.novel_dataset import ObjNovelDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoader, DataLoaderIter
from utils import AverageMeter, parse_devices
from model.parallel.replicate import patch_replication_callback
from model.model_base import ModelBuilder, NovelTuningModule, LearningModule


def train(module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    module.train()

    # main loop
    tic = time.time()
    for i in range(args.train_epoch_iters):
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
                  .format(epoch, i, args.train_epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_feat, args.lr_cls,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.train_epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def validate(module, iterator, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_acc = AverageMeter()

    module.eval()
    # main loop
    tic = time.time()
    for i in range(args.val_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        _, acc = module(batch_data)
        acc = acc.mean()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss and acc
        ave_acc.update(acc.data.item() * 100)

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'Accuracy: {:4.2f}'
                  .format(epoch, i, args.val_epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_acc.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.val_epoch_iters
            history['val']['epoch'].append(fractional_epoch)
            history['val']['acc'].append(acc.data.item())
    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(nets.state_dict(),
               '{}/net_{}'.format(args.ckpt, suffix_latest))


def main(args):
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor(arch=args.arch, weights=args.fe_weight)
    fc_classifier = builder.build_classification_layer(args)

    crit_cls = nn.CrossEntropyLoss(ignore_index=-1)
    crit_seg = nn.NLLLoss(ignore_index=-1)
    crit = [{'type': 'cls', 'crit': crit_cls, 'weight': 1},
            {'type': 'seg', 'crit': crit_seg, 'weight': 0}]

    dataset_train = ObjNovelDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    dataset_val = ObjNovelDataset(
        args.list_val, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_val = DataLoader(
        dataset_val, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )

    args.train_epoch_iters = \
        math.ceil(dataset_train.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    args.val_epoch_iters = \
        math.ceil(dataset_val.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    print('1 Train Epoch = {} iters'.format(args.train_epoch_iters))
    print('1 Val Epoch = {} iters'.format(args.val_epoch_iters))

    iterator_train = iter(loader_train)
    iterator_val = iter(loader_val)

    optimizer_feat = torch.optim.SGD(feature_extractor.parameters(),
                                     lr=args.lr_feat, momentum=0.5)
    optimizer_cls = torch.optim.SGD(fc_classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5)
    # optimizers = [optimizer_feat, optimizer_cls]
    optimizers = [optimizer_cls]
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val': {'epoch': [], 'acc': []}}
    network = NovelTuningModule(feature_extractor, crit, fc_classifier)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    if args.start_epoch != 0:
        network.load_state_dict(
            torch.load('{}/net_epoch_{}.pth'.format(args.ckpt, args.start_epoch - 1)))

    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, history, epoch, args)
        checkpoint(network, history, args, epoch)
        validate(network, iterator_val, history, epoch, args)

    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--fe_weight', default='./weights/feature_16.pth')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Novel/novel_obj_train.odgt')
    parser.add_argument('--list_val',
                        default='./data/ADE/ADE_Novel/novel_obj_val.odgt')
    parser.add_argument('--root_dataset',
                        default='../')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=1000, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_feat', default=5.0 * 1e-2, type=float, help='LR')
    parser.add_argument('--lr_cls', default=5.0 * 1e-2, type=float, help='LR')

    # Data related arguments
    parser.add_argument('--num_class', default=293, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=32, type=int,
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

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./novel_ckpt_1',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()

    main(args)
