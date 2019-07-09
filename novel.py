import os
import time
import random
import argparse
import json

import torch
import torch.nn as nn
from dataset.novel_dataset import ObjNovelDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoader, DataLoaderIter
from utils import AverageMeter, parse_devices
from model.parallel.replicate import patch_replication_callback


def evaluate(module, iterator, history, args, epoch):
    module.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_acc = AverageMeter()

    tic = time.time()

    for i in range(args.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        _, acc = module(batch_data)
        acc = acc.mean()

        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_acc.update(acc.data.item() * 100)

        print("Epoch[{}] Accuracy: {:4.2f}")
        history['test']['epoch'].append(epoch)
        history['test']['acc'].append(ave_acc)



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


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(nets.state_dict(),
               '{}/net_{}'.format(args.ckpt, suffix_latest))


def adjust_learning_rate(optimizers, cur_epoch, args):
    if cur_epoch % 10 == 0 and cur_epoch != 0:
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group - 5 * 1e-3


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor(arch=args.arch)
    fc_classifier = builder.build_classification_layer(args)

    crit_cls = nn.CrossEntropyLoss(ignore_index=-1)
    crit_seg = nn.NLLLoss(ignore_index=-1)
    crit = [{'type': 'cls', 'crit': crit_cls, 'weight': 1},
            {'type': 'seg', 'crit': crit_seg, 'weight': 0}]

    dataset_train = ObjNovelDataset(args.list_train,
                                    args, batch_per_gpu=args.batch_size_per_gpu)
    dataset_test = ObjNovelDataset(args.list_test, args, batch_per_gpu=args.batch_size_per_gpu)

    loader_train = DataLoader(
        dataset_train, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    loader_test = DataLoader(
        dataset_test, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=user_scattered_collate,
        drop_last=True,
        pin_memory=True
    )

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    iterator_train = iter(loader_train)
    iterator_test = iter(loader_test)
    optimizer_feat = torch.optim.SGD(feature_extractor.parameters(),
                                     lr=args.lr_feat, momentum=0.5)
    optimizer_cls = torch.optim.SGD(fc_classifier.parameters(),
                                    lr=args.lr_cls, momentum=0.5)
    optimizers = [optimizer_feat, optimizer_cls]
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'test': {'epoch': [], 'acc': []}}

    network = LearningModule(feature_extractor, crit, fc_classifier)
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()

    if args.start_epoch != 0:
        network.load_state_dict(torch.load('{}/net_epoch_{}.pth'.format(args.ckpt, args.start_epoch - 1)))

    for epoch in range(args.start_epoch, args.num_epoch):
        train(network, iterator_train, optimizers, history, epoch, args)
        checkpoint(network, history, args, epoch)
        if 'test' in args.mode:
            evaluate(network, iterator_test, history, args, epoch)
    print('Training Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='LeNet')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--fe_weight', default=None, help='weight of the feature extractor')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Novel/novel_obj_train.odgt')
    parser.add_argument('--list_test', default='./data/ADE/ADE_Novel/novel_obj_test.odgt')
    parser.add_argument('--root_dataset',
                        default='../')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0, 1],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=10, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_feat', default=1.8 * 1e-2, type=float, help='LR')
    parser.add_argument('--lr_cls', default=1.8 * 1e-2, type=float, help='LR')

    # Data related arguments
    parser.add_argument('--num_class', default=189, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
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
    parser.add_argument('--ckpt', default='./checkpoint_1',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument("--worst_ratio", default=100)
    parser.add_argument("--group_split", default=[1/4, 1 / 2, 1, 2, 4])

    # Mode
    parser.add_argument("--mode", default="train-test", help="training or testing")

    args = parser.parse_args()

    main(args)
