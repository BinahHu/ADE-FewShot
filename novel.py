import os
import time
import random
import argparse
import json
import math
import numpy as np

import torch
import torch.nn as nn
from dataset.novel_dataset import ObjNovelDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoader, DataLoaderIter
from utils import AverageMeter, parse_devices
from model.parallel.replicate import patch_replication_callback
from model.model_base import ModelBuilder, NovelTuningModule

from logger import Logger


def train(module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    module.train()

    # main loop
    tic = time.time()
    acc_iter = 0
    acc_iter_num = 0 
    for i in range(args.train_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        module.zero_grad()
        loss, acc = module(batch_data)
        loss = loss.mean()
        acc = acc.mean()
        acc_iter += acc.data.item() * 100
        acc_iter_num += 1 

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
                  'lr_cls: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, Acc-Iter: {:4.2f}'
                  .format(epoch, i, args.train_epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_cls,
                          ave_acc.average(), ave_total_loss.average(), acc_iter / acc_iter_num))
            info = {'loss-train':ave_total_loss.average(), 'acc-train':ave_acc.average(), 'acc-iter-train': acc_iter / acc_iter_num}
            acc_iter = 0
            acc_iter_num = 0
            dispepoch = epoch
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.train_epoch_iters)

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
    acc_iter = 0
    acc_iter_num = 0
    for i in range(args.val_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        _, acc = module(batch_data)
        acc = acc.mean()
        acc_iter += acc.data.item() * 100
        acc_iter_num += 1

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss and acc
        ave_acc.update(acc.data.item() * 100)

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                    'Accuracy: {:4.2f}, Acc-Iter: {:4.2f}'
                  .format(epoch, i, args.val_epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_acc.average(), acc_iter / acc_iter_num))
                  
            info = {'acc-val':ave_acc.average(), 'acc-iter-val':acc_iter / acc_iter_num}
            acc_iter = 0
            acc_iter_num = 0
            dispepoch = epoch
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.val_epoch_iters)

            fractional_epoch = epoch - 1 + 1. * i / args.val_epoch_iters
            history['val']['epoch'].append(fractional_epoch)
            history['val']['acc'].append(acc.data.item())
    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))
    return ave_acc.average()


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
    classifier = builder.build_classification_layer(args)

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
    
    args.logger = Logger(os.path.join(args.log_dir, args.comment))

    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5)
    optimizers = [optimizer_cls]
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val': {'epoch': [], 'acc': []}}
    network = NovelTuningModule(crit, classifier)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    if args.start_epoch != 0:
        network.load_state_dict(
            torch.load('{}/net_epoch_{}.pth'.format(args.ckpt, args.log)))

    accuracy = []
    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, history, epoch, args)
        accuracy.append(validate(network, iterator_val, history, epoch, args))
        checkpoint(network, history, args, epoch)

    print(np.max(np.array(accuracy)))
    print(np.argmax(np.array(accuracy)))
    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--cls', default='novel_cls')
    parser.add_argument('--feat_dim', default=512)

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/test_feat/img_train_feat.h5')
    parser.add_argument('--list_val',
                        default='./data/test_feat/img_val_feat.h5')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0, 1, 2, 3],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=40, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_cls', default=5.0 * 1e-1, type=float, help='LR')
    parser.add_argument('--weight_init', default='')
    parser.add_argument('--crop_height', default=2)
    parser.add_argument('--crop_width', default=2)

    # Data related arguments
    parser.add_argument('--num_class', default=293, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=8, type=int,
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
    parser.add_argument('--ckpt', default='./ckpt/novel_ckpt/',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='frequency to display')
    parser.add_argument('--log_dir', default="./log_novel/",
                        help='dir to save train and val log')
    parser.add_argument('--comment', default="this_child_may_save_the_world",
                        help='add comment to this test')

    args = parser.parse_args()

    main(args)
