import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Classifier(nn.Module):
    def __init__(self, in_dim, fc_dim, num_class):
        super(FC_Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, num_class)
        # self.fc2 = nn.Linear(fc_dim, num_class)
        self.type = 'fc_cls'

    def forward(self, x):
        x = self.fc1(x)
        return x

    def _acc(self, pred, label, output='dumb'):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        if output == 'dumb':
            return acc
        elif output == 'vis':
            return acc, pred, label