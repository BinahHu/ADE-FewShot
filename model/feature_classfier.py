"""
Given feature map, sampling the required area and output the result
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureClassifier(nn.Module):
    def __init__(self, in_channel, n_class):
        super(FeatureClassifier, self).__init__()
        self.in_channel = in_channel
        self.class_num = n_class

        self.classifier = nn.Sequential(
            nn.Linear(in_channel, n_class),
            nn.ReLU(),
            nn.Linear(in_channel, n_class),
            nn.ReLU(),
            nn.Linear(in_channel, n_class),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, anchor, down_sampling_rate=1):
        """
        compute the classifying result on the anchor area of the image
        :param x: input
        :param anchor: the required area
        :param down_sampling_rate: down sampling of the feature map
        :return: classify result after softmax
        """
        pred = torch.zeros(x.shape[0], self.class_num).to(x.device())
        anchor = (np.array(anchor) / down_sampling_rate).astype(np.int)
        for i in range(x.shape[0]):
            pred[i] = self.classifier(F.max_pool2d(x[i, :, anchor[2]:anchor[3], anchor[0]:anchor[1]]))
        return pred
