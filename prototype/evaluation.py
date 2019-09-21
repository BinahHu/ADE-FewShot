import h5py
import numpy as np
import argparse
import sys
sys.path.append('../')
import utils
import torch


def acc(preds, label, range_of_compute):
    category_acc = np.zeros((2, preds.shape[1]))
    acc_sum = 0
    num = preds.shape[0]
    preds = np.argsort(preds)
    label = label.astype(np.int)
    for i in range(num):
        category_acc[1, label[i]] += 1
        if label[i] in preds[i, -range_of_compute:]:
            acc_sum += 1
            category_acc[0, label[i]] += 1
    acc = np.array(acc_sum / (num + 1e-10))
    cat_acc = utils.category_acc(torch.tensor(category_acc), 1)
    return acc, cat_acc


def main(args):
    train_set = h5py.File(args.list_train, 'r')
    val_set = h5py.File(args.list_val, 'r')
    train_feat = np.array(train_set['feature_map'])
    train_labels = np.array(train_set['labels']).astype(np.int)
    val_feat = np.array(val_set['feature_map'])
    val_labels = np.array(val_set['labels']).astype(np.int)

    # compute the prototype
    instance_num, feat_dim = train_feat.shape
    prototype = np.zeros((args.novel_class, feat_dim))
    for i in range(instance_num):
        feat = train_feat[i]
        label = train_labels[i]
        prototype[label, :] += feat

    prototype /= 5
    print('Prototype Computation Ends')
    print(prototype[:2, :2])

    # evaluate the validation
    instance_num, _ = val_feat.shape
    pred = np.zeros((args.novel_class, instance_num))
    for i in range(args.novel_class):
        proto = np.repeat(prototype[i:i+1, :], instance_num, axis=0)
        delta = np.sum((proto - val_feat) ** 2, axis=1)
        pred[i, :] = delta
        if i % 20 == 0:
            print(i)
    pred = -np.transpose(pred)
    inst_acc1, cat_acc1 = acc(pred, val_labels, 1)
    inst_acc5, cat_acc5 = acc(pred, val_labels, 5)

    print("Top-1 Inst {} Cat {}".format(inst_acc1, cat_acc1))
    print("Top-5 Inst {} Cat {}".format(inst_acc5, cat_acc5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--novel_class', default=100, type=int)
    parser.add_argument('--list_train', default='../data/test_feat/img_train_feat_baseline.h5')
    parser.add_argument('--list_val', default='../data/test_feat/img_val_feat_baseline.h5')

    args = parser.parse_args()
    main(args)
