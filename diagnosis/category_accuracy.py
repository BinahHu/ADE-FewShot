import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import math
from dataset.base_dataset import ObjBaseDataset
from dataset.dataloader import DataLoaderIter, DataLoader
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from utils import AverageMeter
import time

from model.model_base import ModelBuilder, LearningModule
from model.parallel.replicate import patch_replication_callback


def validate(module, iterator, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_acc = AverageMeter()

    module.eval()
    # main loop
    tic = time.time()
    for i in range(args.val_epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        _, acc, pred, label = module(batch_data)
        acc = acc.mean()
        if i == 0:
            preds = np.array(pred.detach().cpu())
            labels = np.array(label.detach().cpu())
        else:
            pred = np.array(pred.detach().cpu())
            label = np.array(label.detach().cpu())
            preds = np.vstack((preds, pred))
            labels = np.hstack((labels, label))

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

    print('Epoch: [{}], Accuracy: {:4.2f}'.format(epoch, ave_acc.average()))
    return preds, labels


def main(args):
    # Network Builders
    builder = ModelBuilder()
    feature_extractor = builder.build_feature_extractor(arch=args.arch, weights=args.model_weight)
    fc_classifier = builder.build_classification_layer(args)

    crit_cls = nn.CrossEntropyLoss(ignore_index=-1)
    crit_seg = nn.NLLLoss(ignore_index=-1)
    crit = [{'type': 'cls', 'crit': crit_cls, 'weight': 1},
            {'type': 'seg', 'crit': crit_seg, 'weight': 0}]


    accuracy = np.zeros(args.num_class)
    dataset_val = ObjBaseDataset(args.list_val, args, batch_per_gpu=args.batch_size_per_gpu)
    loader_val = DataLoader(
        dataset_val, batch_size=len(args.gpus), shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True
    )
    args.val_epoch_iters = \
        math.ceil(dataset_val.num_sample / (args.batch_size_per_gpu * len(args.gpus)))
    print('1 Val Epoch = {} iters'.format(args.val_epoch_iters))

    iterator_val = iter(loader_val)
    network = LearningModule(feature_extractor, crit, fc_classifier, output='vis')
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()
    network.eval()

    preds, labels = validate(network, iterator_val, 1, args)
    labels = np.reshape(labels.size())
    occurence = np.zeros(args.num_class)
    accuracy_top_1 = np.zeros(args.num_class)
    accuracy_top_5 = np.zeros(args.num_class)
    for index, label in labels:
        label = int(label)
        occurence[label] += 1
        pred = preds[index]
        if np.argmax(pred) == label:
            accuracy_top_1[label] += 1
    for i in range(args.num_class):
        accuracy_top_1[i] = accuracy_top_1[i] / occurence[i]

    print('Evaluation Done')
    f = open('top1-accuracy.json', 'w')
    json.dump(accuracy_top_1, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--model_weight', default=None)

    # Path related arguments
    parser.add_argument('--list_train',
                        default='../data/ADE/ADE_Base/base_obj_train.odgt')
    parser.add_argument('--list_val',
                        default='../data/ADE/ADE_Base/base_obj_val.odgt')
    parser.add_argument('--root_dataset',
                        default='../../')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=5, type=int,
                        help='epochs to test')
    parser.add_argument('--val_epoch_iters', default=20, type=int)

    # Data related arguments
    parser.add_argument('--num_class', default=189, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=4, type=int,
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
    parser.add_argument('--ckpt', default='./checkpoint',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument('--sample_type', default='inst')

    args = parser.parse_args()

    main(args)