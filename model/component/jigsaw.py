import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class JigSawClassifier(nn.Module):
    def __init__(self, args):
        super(JigSawClassifier, self).__init__()
        self.num_class = 1000
        self.fc = nn.Linear(9 * 512, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mode = 'train'
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        the input has shape [batch_img_num, 9, C, H, W]
        :param x: input
        :return: loss
        """
        feature, labels = x
        pooled_features = []
        for j in range(9):
            pooled_features.append(self.global_pool(feature[j]))
        concat_feature = torch.cat(pooled_features, 1)
        pred = self.fc(concat_feature)
        loss = self.loss(pred, labels)

        return loss