import torch
from dataset.base_dataset import ImgBaseDataset
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
    feature_extractor_ = builder.build_feature_extractor(arch=args.arch, weights=args.weight_init)
    fc_classifier_ = builder.build_classification_layer(args)
    embed_ = builder.build_embedding_layer(args)
    network_ = LearningModule(args, feature_extractor_, crit=[], embed=embed_, cls=fc_classifier_, output='feat')
    network_ = UserScatteredDataParallel(network_)
    patch_replication_callback(network_)
    network_.load_state_dict(torch.load(args.model))
    torch.save(network_.module.state_dict(), 'tmp.pth')

    print('Real Loading Start')
    feature_extractor = builder.build_feature_extractor(arch=args.arch)
    fc_classifier = builder.build_classification_layer(args)
    embed = builder.build_embedding_layer(args)
    network = LearningModule(args, feature_extractor, crit=[], embed=embed, cls=fc_classifier, output='feat')
    network.load_state_dict(torch.load('tmp.pth'))
    network = UserScatteredDataParallel(network, device_ids=args.gpus)
    patch_replication_callback(network)
    network.cuda()
    network.eval()

    dataset_train = ImgBaseDataset(args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    dataset_train.mode = 'val'
    dataset_val = ImgBaseDataset(args.list_val, args, batch_per_gpu=args.batch_size_per_gpu)
    dataset_val.mode = 'val'
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

    iterations = 0
    while iterations <= args.train_epoch_iters:
        batch_data = next(iter_train)
        if iterations % 10 == 0:
            print('{} / {}'.format(iterations, args.train_epoch_iters))
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())
        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
        iterations += 1

    f = h5py.File('data/test_feat/img_train_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.close()

    iterations = 0
    while iterations <= args.val_epoch_iters:
        batch_data = next(iter_val)
        if iterations % 10 == 0:
            print('{} / {}'.format(iterations, args.val_epoch_iters))
        if iterations == 0:
            features, labels = network(batch_data)
            features = np.array(features.detach().cpu())
            labels = np.array(labels.cpu())
            anchors = np.array(batch_data[0]['anchors'][0, :labels.size, :])
            scales = np.tile(np.array(batch_data[0]['scales']), (labels.size, 1))

        else:
            feature, label = network(batch_data)
            feature = np.array(feature.detach().cpu())
            label = np.array(label.cpu())
            anchor = np.array(batch_data[0]['anchors'][0, :label.size, :])
            scale = np.tile(np.array(batch_data[0]['scales'][:label.size, :]), (label.size, 1))

            features = np.vstack((features, feature))
            labels = np.hstack((labels, label))
            anchors = np.vstack((anchors, anchor))
            scales = np.vstack((scales, scale))
        iterations += 1

    f = h5py.File('data/test_feat/img_val_feat.h5', 'w')
    f.create_dataset('feature_map', data=features)
    f.create_dataset('labels', data=labels)
    f.create_dataset('anchors', data=anchors)
    f.create_dataset('scales', data=scales)
    f.close()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--cls', default='linear')
    parser.add_argument('--feat_dim', default=512)
    parser.add_argument('--crop_height', default=3)
    parser.add_argument('--crop_width', default=3)
    parser.add_argument('--loss', default='CE')

    parser.add_argument('--num_attr', default=386, type=int)
    parser.add_argument('--is_soft', default=False, help='use soft attrinute loss')
    parser.add_argument('--attr_weight', default=0.1, type=float, help='Weight of the attribute loss')
    parser.add_argument('--orth_weight', default=0, type=float, help='Weight of the orthogonality')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE/ADE_Novel/novel_img_train.json')
    parser.add_argument('--list_val',
                        default='./data/ADE/ADE_Novel/novel_img_val.json')
    parser.add_argument('--root_dataset',
                        default='../')
    parser.add_argument('--attr_path', default='data/ADE/ADE_Origin/attr.json')

    # optimization related arguments
    parser.add_argument('--gpus', default=[0],
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--train_epoch_iters', default=20, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--val_epoch_iters', default=20, type=int)
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--weight_init', default='')

    # Data related arguments
    parser.add_argument('--num_class', default=189, type = int)
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
    parser.add_argument('--sample_per_img', default=-1)


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
    parser.add_argument('--model', default="ckpt/crop_res_50/net_epoch_11.pth",
                        help='model to load')
    parser.add_argument('--max_anchor_per_img', default=100)

    args = parser.parse_args()

    save_feature(args)