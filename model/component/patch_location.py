import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchLocationClassifier(nn.Module):
    def __init__(self, args):
        super(PatchLocationClassifier, self).__init__()
        self.num_class = 8
        self.fc = nn.Linear(2 * 512, self.num_class)
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
        batch_img_num, _, channel, _, _ = feature.shape
        pooled_features = []
        for j in range(2):
            pooled_features.append(self.global_pool(feature[:, j, :, :, :]))
        concat_feature = torch.cat(pooled_features, 1).view(batch_img_num, -1)
        pred = self.fc(concat_feature)
        loss = self.loss(pred, labels)

        return loss
