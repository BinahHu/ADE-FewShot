import os
import time
import random
import argparse
import json
import math
import numpy as np

import torch
import torch.nn as nn
from dataset.novel_dataset import NovelDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoader, DataLoaderIter
from utils import AverageMeter, category_acc
from model.parallel.replicate import patch_replication_callback
from model.novel_model import NovelClassifier, NovelCosClassifier
import copy


def train(module, iterator, optimizers, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    module.train()
    module.module.mode = 'train'
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
            acc_iter = 0
            acc_iter_num = 0


def validate(module, iterator, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_acc = AverageMeter()

    module.eval()
    module.module.mode = 'val'
    # main loop
    tic = time.time()
    acc_iter = 0
    acc_iter_num = 0
    category_accuracy = torch.zeros(2, args.num_novel_class)
    for i in range(args.val_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        acc, category_batch_acc = module(batch_data)
        acc = acc.mean()
        acc_iter += acc.data.item() * 100
        acc_iter_num += 1
        category_batch_acc = category_batch_acc.cpu()
        # print(category_batch_acc[:, :10])
        for j in range(len(args.gpus)):
            category_accuracy += category_batch_acc[2*j:2*j+2, :]

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

            acc_iter = 0
            acc_iter_num = 0
    # print(category_accuracy)
    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))
    acc = category_acc(category_accuracy, args)
    print('Ave Category Acc: {:4.2f}'.format(acc.item() * 100))
    return [ave_acc.average(), acc]


def checkpoint(nets, args, epoch_num):
    print('Saving checkpoints...')
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(nets.module.state_dict(),
               '{}/net_{}'.format(args.ckpt, suffix_latest))


def slide_window_ave(acc_list, window_size=10):
    category = []
    inst = []
    for sample in acc_list:
        category.append(sample[1])
        inst.append(sample[0])
    category = np.array(category)
    inst = np.array(inst)
    epoch = category.size

    start_location = 0
    best_shot = -1
    for i in range(0, epoch - window_size):
        cur_value = category[i:i+window_size].mean()
        if cur_value > best_shot:
            start_location = i
            best_shot = cur_value

    best_inst = inst[start_location:start_location+window_size].mean()
    print('Best Category {}'.format(best_shot))
    print('Best Inst {}'.format(best_inst))


def main(args):
    dataset_train = NovelDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )

    vargs = copy.deepcopy(args)
    vargs.gpus = [0, 1, 2, 3]
    vargs.batch_size_per_gpu = 213
    vargs.disp_iter = 1
    dataset_val = NovelDataset(
        args.list_val, args, batch_per_gpu=vargs.batch_size_per_gpu)
    loader_val = DataLoader(
        dataset_val, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )

    args.train_epoch_iters = \
        math.ceil(dataset_train.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    vargs.val_epoch_iters = \
        math.ceil(dataset_val.num_sample / (vargs.batch_size_per_gpu * len(vargs.gpus)))
    print('1 Train Epoch = {} iters'.format(args.train_epoch_iters))
    print('1 Val Epoch = {} iters'.format(vargs.val_epoch_iters))

    iterator_train = iter(loader_train)
    iterator_val = iter(loader_val)

    if args.cls == 'novel_cls':
        classifier = NovelClassifier(args)
    elif args.cls == 'novel_coscls':
        classifier = NovelCosClassifier(args)
    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5)
    optimizers = [optimizer_cls]
    network = UserScatteredDataParallel(classifier, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    if args.start_epoch != 0:
        network.load_state_dict(
            torch.load('{}/net_epoch_{}.pth'.format(args.ckpt, args.log)))

    accuracy = []
    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, epoch, args)
        accuracy.append(validate(network, iterator_val, epoch, vargs))
        # checkpoint(network, args, epoch)

    slide_window_ave(accuracy)
    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet10')
    parser.add_argument('--cls', default='novel_cls')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--crop_height', default=3, type=int)
    parser.add_argument('--crop_width', default=3, type=int)
    parser.add_argument('--range_of_compute', default=5, type=int)

    # Path related arguments
    parser.add_argument('--list_train',
                        default='data/test_feat/img_train_feat_comb2.h5')
    parser.add_argument('--list_val',
                        default='data/test_feat/img_val_feat_comb2.h5')
    # optimization related arguments
    parser.add_argument('--gpus', default=[0, 1, 2, 3],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_cls', default=5.0 * 1e-1, type=float, help='LR')
    parser.add_argument('--weight_init', default='')

    # Data related arguments
    parser.add_argument('--num_novel_class', default=100, type=int,
                        help='number of classes')
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
