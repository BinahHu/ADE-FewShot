import sys
sys.path.append('../')
import h5py
import numpy as np
import argparse
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
    models = args.models
    preds = []
    labels = None
    for model in models:
        file_path = 'pred/img_test_pred_{}.h5'.format(model)
        f = h5py.File(file_path, 'r')
        preds.append(np.array(f['preds']))
        if labels is None:
            labels = np.array(f['labels'])
        else:
            if not np.array_equal(labels, np.array(f['labels'])):
                raise RuntimeError('mismatch list')

    model_num = len(models)
    instance_num = preds[0].shape[0]
    class_num = preds[0].shape[1]
    weight = args.weight

    pred_vote = np.zeros((instance_num, class_num))
    for i in range(model_num):
        pred_vote += weight[i] * preds[i]

    acc_1 = acc(pred_vote, labels, range_of_compute=1)
    acc_5 = acc(pred_vote, labels, range_of_compute=5)

    print(acc_1)
    print(acc_5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=['baseline'])
    parser.add_argument('-weight', default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--mode', default='val')
    args = parser.parse_args()
    main(args)
