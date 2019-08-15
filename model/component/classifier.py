import torch
import torch.nn as nn
import logging


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.num_class = args.num_base_class
        self.down_sampling_rate = args.down_sampling_rate
        self.fc = nn.Linear(args.feat_dim * args.crop_width * args.crop_height, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mode = 'train'

    def _acc(self, pred, label):
        category_accuracy = torch.zeros(2, self.num_class)
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        for i, label_instance in enumerate(label):
            category_accuracy[1, label_instance] += 1
            if preds[i] == label_instance:
                category_accuracy[0, label_instance] += 1
        del pred
        return acc, category_accuracy

    def diagnosis(self, x):
        """
        used only when diagnosis
        :param x: input
        :return: prediction
        """
        feature, labels = x
        pred = self.fc(feature)
        return pred

    def forward(self, x):
        if self.mode == 'diagnosis':
            return self.diagnosis(x)
        loss = 0
        feature, labels = x
        feat_layers = len(feature)
        pred = None
        for i in range(feat_layers - 1, feat_layers):
            pred = self.fc(feature[i])
            loss += self.loss(pred, labels)
        acc, category_accuracy = self._acc(pred, labels)

        return loss, acc, category_accuracy
