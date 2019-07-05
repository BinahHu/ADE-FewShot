"""
Prototype for the whole network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_classfier import FeatureClassifier
from model.fcn import fcn8s, fcn16s


class NetWork(nn.Module):
    def __init__(self, n_class, feature_channel):
        super(NetWork, self).__init__()
        self.n_class = n_class
        self.feature_channel = feature_channel
        self.backbone = fcn16s(feature_channel)
        self.classifier = FeatureClassifier(feature_channel, n_class)

    def forward(self, x, anchor, down_sampling_rate=1):
        feature = self.backbone(x)
        result = self.classifier(feature)
        return result
