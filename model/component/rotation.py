import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotationClassifier(nn.Module):
    def __init__(self, args):
        super(RotationClassifier, self).__init__()
        self.num_class = 4
        self.fc = nn.Linear(512, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature, labels = x
        pooled_feature = self.global_pool(feature)
        pred = self.fc(pooled_feature.view(-1, 512))
        loss = self.loss(pred, labels)

        return loss
