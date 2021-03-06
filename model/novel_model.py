import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F


class NovelClassifier(nn.Module):
    def __init__(self, args):
        super(NovelClassifier, self).__init__()
        self.in_dim = args.crop_height * args.crop_width * args.feat_dim
        self.num_class = args.num_novel_class
        self.fc = nn.Linear(self.in_dim, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mode = 'train'

        if hasattr(args, 'range_of_compute'):
            self.range_of_compute = args.range_of_compute

    def predict(self, x):
        feature = x['feature']
        label = x['label'].long()
        pred = self.fc(feature)
        return self.acc(pred, label)

    def probability(self, x):
        feature = x['feature']
        label = x['label'].long()
        pred = self.fc(feature)
        prob = F.softmax(pred)
        return prob, label

    def forward(self, x):
        if self.mode == 'val':
            return self.predict(x)
        elif self.mode == 'diagnosis':
            return self.diagnosis(x)
        elif self.mode == 'prob':
            return self.probability(x)

        feature = x['feature']
        label = x['label'].long()
        pred = self.fc(feature)
        loss = self.loss(pred, label)
        acc = self._acc(pred, label)
        return loss, acc

    def _acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        return acc

    def acc(self, pred, label):
        category_acc = torch.zeros(2, self.num_class).cuda()
        acc_sum = 0
        num = pred.shape[0]
        preds = np.array(pred.detach().cpu())
        preds = np.argsort(preds)
        label = np.array(label.detach().cpu())
        for i in range(num):
            category_acc[1, label[i]] += 1
            if label[i] in preds[i, -self.range_of_compute:]:
                acc_sum += 1
                category_acc[0, label[i]] += 1
        acc = torch.tensor(acc_sum / (num + 1e-10)).cuda()
        return acc, category_acc


class NovelCosClassifier(nn.Module):
    def __init__(self, args):
        super(NovelCosClassifier, self).__init__()
        self.in_dim = args.crop_height * args.crop_width * args.feat_dim
        self.num_class = args.num_novel_class
        self.fc = nn.Linear(self.in_dim, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mode = 'train'

        self.t = torch.ones(1).cuda() * 10
        self.weight = nn.Parameter(torch.Tensor(self.num_class, self.in_dim))
        self.reset_parameters()

        if hasattr(args, 'range_of_compute'):
            self.range_of_compute = args.range_of_compute

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def predict(self, x):
        feature = x['feature']
        label = x['label'].long()
        xx = feature
        batch_size = xx.size(0)
        pred = self.t.cuda() * cosine_similarity(
            xx.unsqueeze(1).expand(batch_size, self.num_class, self.in_dim),
            self.weight.unsqueeze(0).expand(batch_size, self.num_class, self.in_dim).cuda(), 2)
        return self.acc(pred, label)

    def forward(self, x):
        if self.mode == 'val':
            return self.predict(x)
        elif self.mode == 'diagnosis':
            return self.diagnosis(x)

        feature = x['feature']
        label = x['label'].long()

        xx = feature
        batch_size = xx.size(0)
        pred = self.t.cuda() * cosine_similarity(
            xx.unsqueeze(1).expand(batch_size, self.num_class, self.in_dim),
            self.weight.unsqueeze(0).expand(batch_size, self.num_class, self.in_dim).cuda(), 2)
        loss = self.loss(pred, label)

        acc = self._acc(pred, label)
        return loss, acc

    def _acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        return acc

    def acc(self, pred, label):
        category_acc = torch.zeros(2, self.num_class).cuda()
        acc_sum = 0
        num = pred.shape[0]
        preds = np.array(pred.detach().cpu())
        preds = np.argsort(preds)
        label = np.array(label.detach().cpu())
        for i in range(num):
            category_acc[1, label[i]] += 1
            if label[i] in preds[i, -self.range_of_compute:]:
                acc_sum += 1
                category_acc[0, label[i]] += 1
        acc = torch.tensor(acc_sum / (num + 1e-10)).cuda()
        return acc, category_acc
