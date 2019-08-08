import os
import time
import random
import argparse
import json
import math
import copy

import torch
import torch.nn as nn

from dataset.base_dataset import BaseDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoaderIter, DataLoader
from utils import AverageMeter, parse_devices

from model.builder import ModelBuilder
from model.base_model import BaseLearningModule
from model.parallel.replicate import patch_replication_callback

from utils import selective_load_weights

from logger import Logger


def train(module, iterator, optimizers, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    if len(args.supervision) != 0:
        ave_loss_cls = AverageMeter()
        ave_supervision_loss = []
        for supervision in args.supervision:
            ave_supervision_loss.append(AverageMeter())

    module.train()
    module.module.mode = 'train'
    # main loop
    tic = time.time()
    acc_disp = 0
    inst_disp = 0
    for i in range(args.train_epoch_iters):
        if args.isWarmUp is True:
            warm_up_adjust_lr(optimizers, epoch, i, args)
        else:
            train_adjust_lr(optimizers, epoch, i, args)

        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        module.zero_grad()
        loss, acc, instances, loss_cls, loss_supervision = module(batch_data)
        instances = instances.type_as(acc).detach()
        acc = acc.detach()
        loss = (loss * instances).sum() / instances.sum().float()
        acc_actual = (acc * instances).sum() / instances.sum().float()
        acc_disp += (acc * instances).sum()
        inst_disp += instances.sum()

        # print(loss_cls)
        # print(loss_supervision)
        if loss_cls is not None:
            loss_cls = (loss_cls * instances).sum() / instances.sum().float()
            loss_supervision = (loss_supervision * instances).sum() / instances.sum().float()

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
            if loss_cls is not None:
                ave_loss_cls.update(loss_cls.item())
                for j in range(len(args.supervision)):
                    ave_supervision_loss[j].update(loss_supervision)

        if i % args.display_iter == 0:
            message = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, ' \
                      'lr_feat: {:.6f}, lr_cls: {:.6f}, Accuracy: {:4.2f}, ' \
                      'Loss: {:.6f}, Acc-Iter: {:4.2f}, '.format(epoch, i, args.train_epoch_iters, batch_time.average(),
                                                               data_time.average(), optimizers[0].param_groups[0]['lr'],
                                                               optimizers[1].param_groups[0]['lr'], ave_acc.average(),
                                                               ave_total_loss.average(), acc_disp / inst_disp * 100)
            info = {'loss-train': ave_total_loss.average(), 'acc-train': ave_acc.average(),
                    'acc-iter-train': acc_disp / inst_disp * 100}
            acc_disp = 0
            inst_disp = 0
            if loss_cls is not None:
                message += 'Loss_Cls: {:.6f}, '.format(ave_loss_cls.average())
                info['loss-cls'] = ave_loss_cls.average()
                for j in range(len(args.supervision)):
                    message += 'Loss-{}: {:.6f}, '.format(args.supervision[j]['name'],
                                                          ave_supervision_loss[j].average())
                    info['loss-' + args.supervision[j]['name']] = ave_supervision_loss[j].average()

            print(message)

            dispepoch = epoch
            if not args.isWarmUp:
                dispepoch += 1
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.train_epoch_iters)

        del loss
        del acc_actual
        del instances
        del batch_data


def validate(module, iterator, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_acc = AverageMeter()
    ave_total_loss = AverageMeter()

    module.eval()
    module.module.mode = 'val'
    # main loop
    tic = time.time()
    acc_disp = 0
    inst_disp = 0
    for i in range(args.val_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        loss, acc, instances = module(batch_data)
        instances = instances.type_as(acc).detach().cpu()
        acc = acc.detach().cpu()
        acc_actual = (acc * instances).sum() / instances.sum().float()
        loss = (loss.detach().cpu() * instances).sum() / instances.sum().float()
        acc_disp += (acc * instances).sum()
        inst_disp += instances.sum()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss and acc
        for k in range(int(instances.sum())):
            ave_acc.update(acc_actual.data.item() * 100)
            ave_total_loss.update(loss.item())

        if i % args.display_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, Acc-Iter: {:4.2f}'
                  .format(epoch, i, args.val_epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_acc.average(), ave_total_loss.average(), acc_disp / inst_disp * 100))
            
            info = {'loss_val': ave_total_loss.average(), 'acc-val': ave_acc.average(), 'acc-iter-val': acc_disp / inst_disp * 100}
            acc_disp = 0
            inst_disp = 0
            dispepoch = epoch
            if not args.isWarmUp:
                dispepoch += 1
            for tag, value in info.items():
                args.logger.scalar_summary(tag, value, i + dispepoch * args.val_epoch_iters)
        del batch_data
    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))


def checkpoint(nets, args, epoch_num):
    print('Saving checkpoints to {}...'.format(args.ckpt))
    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(nets.module.state_dict(),
               '{}/net_{}'.format(args.ckpt, suffix_latest))


def warm_up_adjust_lr(optimizers, epoch, iteration, args):
    for optimizer in optimizers:
        lr = args.lr_feat * args.warm_up_factor
        lr = lr + (args.lr_feat - lr) * \
             (epoch * args.train_epoch_iters + iteration) / args.warm_up_iters
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_adjust_lr(optimizers, epoch, iteration, args):
    if iteration == 0 and epoch in args.drop_point:
        times = args.drop_point.index(epoch)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / pow(10, times)
    return None


def main(args):
    torch.backends.cudnn.deterministic = True
    # Network Builders
    builder = ModelBuilder(args)
    feature_extractor = builder.build_backbone()
    classifier = builder.build_classifier()

    # supervision
    supervision_modules = []
    for supervision in args.supervision:
        supervision_modules.append({'name': supervision['name'],
                                    'module': getattr(builder, 'build_' + supervision['name'])()})
    setattr(args, 'module', supervision_modules)

    dataset_train = BaseDataset(args.list_train, args)
    dataset_train.mode = 'train'
    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    dataset_val = BaseDataset(args.list_val, args)
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
    # supervision optimizers
    for i, supervision in enumerate(args.supervision):
        optimizers.append(torch.optim.SGD(
            supervision_modules[i]['module'].parameters(),
            lr=supervision['lr'], momentum=0.5, weight_decay=args.weight_decay))

    network = BaseLearningModule(args, backbone=feature_extractor, classifier=classifier)
    if args.model_weight != '':
        selective_load_weights(network, args.model_weight)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    args.isWarmUp = False
    args.logger = Logger(os.path.join(args.log_dir, args.comment))
    # warm up
    if args.log == '' and args.start_epoch == 0:
        print('Start Warm Up')
        args.isWarmUp = True
        args.warm_up_iters = args.warm_up_epoch * args.train_epoch_iters
        for warm_up_epoch in range(args.warm_up_epoch):
            train(network, iterator_train, optimizers, warm_up_epoch, args)
            validate(network, iterator_val, warm_up_epoch, args, )
            checkpoint(network, args, -args.warm_up_epoch + warm_up_epoch)

    args.isWarmUp = False

    # train for real
    optimizer_feat = torch.optim.SGD(feature_extractor.parameters(),
                                     lr=args.lr_feat, momentum=0.5, weight_decay=args.weight_decay)
    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5, weight_decay=args.weight_decay)
    optimizers = [optimizer_feat, optimizer_cls]
    # supervision optimizers
    for i, supervision in enumerate(args.supervision):
        optimizers.append(torch.optim.SGD(
            getattr(network.module, supervision['name']).parameters(),
            lr=supervision['lr'], momentum=0.5, weight_decay=args.weight_decay))

    if args.start_epoch != 0:
        times = 0
        for i in args.drop_point:
            if args.start_epoch > i:
                times += 1
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / pow(10, times)

    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, epoch, args)
        validate(network, iterator_val, epoch, args)
        checkpoint(network, args, epoch)
        torch.cuda.empty_cache()

    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--architecture', default='resnet18')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--crop_height', default=1, type=int)
    parser.add_argument('--crop_width', default=1, type=int)
    parser.add_argument('--model_weight', default='')
    parser.add_argument('--log', default='', help='load trained checkpoint')
    parser.add_argument('--num_base_class', default=189, type=int, help='number of classes')
    parser.add_argument('--padding_constant', default=8, type=int, help='max down sampling rate of the network')
    parser.add_argument('--down_sampling_rate', default=8, type=int, help='down sampling rate')

    # data loading arguments
    parser.add_argument('--supervision', default='supervision.json', type=str)
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Base/base_img_train.json')
    parser.add_argument('--list_val',
                        default='./data/ADE/ADE_Base/base_img_val.json')
    parser.add_argument('--root_dataset', default='../')
    parser.add_argument('--drop_point', default=[13], type=list)
    parser.add_argument('--max_anchor_per_img', default=100)
    parser.add_argument('--workers', default=8, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgShortSize', default=800, type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1500, type=int,
                        help='maximum input image size of long edge')

    # running arguments
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='input batch size')
    parser.add_argument('--num_epoch', default=12, type=int, help='epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_feat', default=1.0 * 1e-3, type=float, help='LR')
    parser.add_argument('--lr_cls', default=1.0 * 1e-3, type=float, help='LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # warm up
    parser.add_argument('--warm_up_epoch', type=int, default=1)
    parser.add_argument('--warm_up_factor', default=0.0001)
    parser.add_argument('--warm_up_iters', default=100)

    # logging
    parser.add_argument('--ckpt', default='./checkpoint', help='folder to output checkpoints')
    parser.add_argument('--display_iter', type=int, default=10, help='frequency to display')
    parser.add_argument('--log_dir', default="./log_base/", help='dir to save train and val log')
    parser.add_argument('--comment', default="this_child_may_save_the_world", help='add comment to this train')

    args = parser.parse_args()

    if args.supervision != '':
        args.supervision = json.load(open(args.supervision, 'r'))
        print(args.supervision)
    else:
        args.supervision = []

    if args.log != '':
        args.model_weight = args.ckpt + 'net_epoch_' + args.log + '.pth'
    main(args)
