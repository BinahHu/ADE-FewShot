import os
import time
import random
import argparse
import json
import math
import copy

import torch
import torch.nn as nn

from dataset.base_dataset import ImgBaseDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoaderIter, DataLoader
from utils import  AverageMeter, parse_devices

from model.model_base import ModelBuilder, LearningModule
from model.parallel.replicate import patch_replication_callback

from loss.focal import FocalLoss

from logger import Logger


def train(module, iterator, optimizers, history, epoch, args, mode='warm'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    module.train()

    # main loop
    tic = time.time()
    for i in range(args.train_epoch_iters):
        if mode=='warm':
            warm_up_adjust_lr(optimizers, epoch, i, args)
        elif mode=='train':
            train_adjust_lr(optimizers, epoch, i, args)

        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        module.zero_grad()
        loss, acc, instances = module(batch_data)
        # print(loss)
        instances = instances.type_as(acc).detach()
        acc = acc.detach()
        loss = (loss * instances).sum() / instances.sum().float()
        acc_actual = (acc * instances).sum() / instances.sum().float()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        for k in range(int(instances.sum())):
            ave_total_loss.update(loss.data.item())
            ave_acc.update(acc_actual * 100)

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_feat: {:.6f}, lr_cls: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, Acc-Iter: {:4.2f}'
                  .format(epoch, i, args.train_epoch_iters,
                          batch_time.average(), data_time.average(),
                          optimizers[0].param_groups[0]['lr'], optimizers[1].param_groups[0]['lr'],
                          ave_acc.average(), ave_total_loss.average(), acc_actual * 100))
            info = {'loss-train':ave_total_loss.average(), 'acc-train':ave_acc.average(), 'acc-iter-train': acc_actual * 100}
            dispepoch = epoch
            if not args.iswarmup:
                dispepoch += 1
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.train_epoch_iters)


            fractional_epoch = epoch - 1 + 1. * i / args.train_epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(ave_acc.average)
        del loss
        del acc_actual
        del instances
        del batch_data


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

        _, acc, instances = module(batch_data)
        instances = instances.type_as(acc).detach().cpu()
        acc = acc.detach().cpu()
        acc_actual = (acc * instances).sum() / instances.sum().float()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss and acc
        for k in range(int(instances.sum())):
            ave_acc.update(acc_actual.data.item() * 100)

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                    'Accuracy: {:4.2f}, Acc-Iter: {:4.2f}'
                  .format(epoch, i, args.val_epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_acc.average(), acc_actual * 100))
            
            info = {'acc-val':ave_acc.average(), 'acc-iter-val':acc_actual * 100}
            dispepoch = epoch
            if not args.iswarmup:
                dispepoch += 1
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.val_epoch_iters)

            fractional_epoch = epoch - 1 + 1. * i / args.val_epoch_iters
            history['val']['epoch'].append(fractional_epoch)
            history['val']['acc'].append(ave_acc.average())
        del batch_data
    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints to {}...'.format(args.ckpt))
    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(nets.state_dict(),
               '{}/net_{}'.format(args.ckpt, suffix_latest))


def warm_up_adjust_lr(optimizers, epoch, iteration, args):
    # print('Adjust learning rate in warm up')
    for optimizer in optimizers:
        lr = args.lr_feat * args.warm_up_factor
        lr = lr + (args.lr_feat - lr) * \
             (epoch * args.train_epoch_iters + iteration) / args.warm_up_iters
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_adjust_lr(optimizers, epoch, iteration, args):
    if (epoch == 3 or epoch == 6 or epoch == 9) and iteration == 0:
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10
    return None


def main(args):
    torch.backends.cudnn.deterministic = True
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor(arch=args.arch, weights=args.weight_init)
    classifier = builder.build_classification_layer(args)

    if args.loss == 'CE':
        crit_cls = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'Focal':
        crit_cls = FocalLoss(class_num = args.num_class, dev_num = len(args.gpus))
    else:
        crit_cls = nn.CrossEntropyLoss(ignore_index=-1)

    crit_seg = nn.NLLLoss(ignore_index=-1)
    crit = [{'type': 'cls', 'crit': crit_cls, 'weight': 1},
            {'type': 'seg', 'crit': crit_seg, 'weight': 0}]

    dataset_train = ImgBaseDataset(
        args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    dataset_train.mode = 'val'
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    valargs = copy.deepcopy(args)
    valargs.sample_type = 'inst'    # always use instance level sampling on val set
    dataset_val = ImgBaseDataset(
        args.list_val, valargs, batch_per_gpu=args.batch_size_per_gpu)
    dataset_val.mode = 'val'
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
                                     lr=2.0 * 1e-4, momentum=0.5, weight_decay=args.weight_decay)
    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=2.0 * 1e-4, momentum=0.5, weight_decay=args.weight_decay)
    optimizers = [optimizer_feat, optimizer_cls]
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val': {'epoch': [], 'acc': []}}

    network = LearningModule(args, feature_extractor, crit, classifier)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    if args.log != '':
        network.load_state_dict(
            torch.load('{}/net_epoch_{}.pth'.format(args.ckpt, args.log)))
        history = torch.load('{}/history_epoch_{}.pth'.format(args.ckpt, args.log))
    
    args.logger = Logger(os.path.join(args.log_dir, args.comment))

    args.iswarmup = False

    # warm up
    if args.log == '' and args.start_epoch  == 0:
        print('Start Warm Up')
        args.iswarmup = True
        args.warm_up_iters = args.warm_up_epoch * args.train_epoch_iters
        for warm_up_epoch in range(args.warm_up_epoch):
            train(network, iterator_train, optimizers, history, warm_up_epoch, args)
            validate(network, iterator_val, history, warm_up_epoch, args, )
            checkpoint(network, history, args, -args.warm_up_epoch + warm_up_epoch)
        history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val': {'epoch': [], 'acc': []}}
    args.iswarmup = False

    # train for real
    optimizer_feat = torch.optim.SGD(feature_extractor.parameters(),
                                     lr=args.lr_feat, momentum=0.5, weight_decay=args.weight_decay)
    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5, weight_decay=args.weight_decay)
    optimizers = [optimizer_feat, optimizer_cls]
    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, history, epoch, args, mode='train')
        validate(network, iterator_val, history, epoch, args)
        checkpoint(network, history, args, epoch)
        torch.cuda.empty_cache()

    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--cls', default='linear')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--log', default='', help='load trained checkpoint')
    parser.add_argument('--loss', default='CE', help='specific the training loss')
    parser.add_argument('--crop_height', default=3)
    parser.add_argument('--crop_width', default=3)

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Base/base_img_train.json')
    parser.add_argument('--list_val',
                        default='./data/ADE/ADE_Base/base_img_val.json')
    parser.add_argument('--root_dataset',
                        default='../')

    # Train parameters
    parser.add_argument('--max_anchor_per_img', default=100)
    parser.add_argument('--sample_per_img', default=-1)

    # optimization related arguments
    parser.add_argument('--gpus', default=[0, 1, 2, 3],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=2, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_feat', default=1.0 * 1e-1, type=float, help='LR')
    parser.add_argument('--lr_cls', default=1.0 * 1e-1, type=float, help='LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--weight_init', default='')

    # Warm up
    parser.add_argument('--warm_up_epoch', type=int, default=1)
    parser.add_argument('--warm_up_factor', default=0.001)
    parser.add_argument('--warm_up_iters', default=100)

    # Data related arguments
    parser.add_argument('--num_class', default=189, type=int,
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
    parser.add_argument('--comment', default="this_child_may_save_the_world",
                        help='add comment to this train')

    args = parser.parse_args()

    main(args)
