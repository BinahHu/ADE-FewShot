import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math


class MaskPredictor(nn.Module):
    def __init__(self, args):
        super(MaskPredictor, self).__init__()
        self.in_dim = args.feat_dim
        self.args = args
        self.down_sampling_rate = args.down_sampling_rate

        self.fc1 = nn.Conv2d(self.in_dim, self.args.num_base_class + 1, kernel_size=3, stride=1, padding=1)

        self.base_classes = json.load(open('data/ADE/ADE_Origin/base_list.json', 'r'))

    @staticmethod
    def compute_anchor_location(anchor, scale, original_scale):
        anchor = np.array(anchor.detach().cpu())
        original_scale = np.array(original_scale)
        scale = np.array(scale.cpu())
        anchor[:, 2] = np.floor(anchor[:, 2] * scale[0] * original_scale[0])
        anchor[:, 3] = np.ceil(anchor[:, 3] * scale[0] * original_scale[0])
        anchor[:, 0] = np.floor(anchor[:, 0] * scale[1] * original_scale[1])
        anchor[:, 1] = np.ceil(anchor[:, 1] * scale[1] * original_scale[1])
        return anchor.astype(np.int)

    @staticmethod
    def binary_transform(mask, label):
        return mask[:, int(label.item()), :, :]

    def forward(self, agg_input):
        """
        take in the feature map and make predictions
        :param agg_input: input data
        :return: loss averaged over instances
        """
        feature_map = agg_input['feature_map']
        mask = agg_input['seg']
        feature_map = feature_map.unsqueeze(0)
        predicted_map = self.fc1(feature_map)
        predicted_map = F.interpolate(predicted_map, size=(mask.shape[0], mask.shape[1]), mode='nearest')
        mask = mask.unsqueeze(0)
        weight = torch.ones(self.args.num_base_class + 1).cuda()
        weight[-1] = 0.1

        loss = F.cross_entropy(predicted_map, mask.long(), weight=weight)
        return loss
