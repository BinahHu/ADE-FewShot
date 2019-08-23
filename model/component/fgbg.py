import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2


class FGBGMaskPredictor(nn.Module):
    def __init__(self, args):
        super(FGBGMaskPredictor, self).__init__()
        self.in_dim = args.feat_dim
        self.args = args
        self.down_sampling_rate = args.down_sampling_rate

        self.fc1 = nn.Conv2d(self.in_dim, self.args.num_base_class + 1, kernel_size=3, stride=1, padding=1)
        # self.fc2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.base_classes = json.load(open('data/ADE/ADE_Origin/base_list.json', 'r'))

        self.diagnosis_count = 0

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

    def diagnosis(self, agg_input):
        raise NotImplementedError

    def forward(self, agg_input):
        """
        take in the feature map and make predictions
        :param agg_input: input data
        :return: loss averaged over instances
        """
        if self.mode == 'diagnosis':
            self.diagnosis(agg_input)
        feature_map = agg_input['feature_map']
        mask = agg_input['seg']

        feature_map = feature_map.unsqueeze(0)
        predicted_map = self.fc1(feature_map)
        predicted_map = F.interpolate(predicted_map, size=(mask.shape[0], mask.shape[1]))

        ones_map = torch.ones(mask.shape[0], mask.shape[1])
        zeros_map = torch.zeros(mask.shape[0], mask.sahpe[1])
        tgt_mask = torch.where(mask == 255, ones_map, zeros_map)
        weight_map = torch.where(0 < mask < 255, zeros_map, ones_map)
        loss = F.binary_cross_entropy_with_logits(predicted_map, tgt_mask.unsqueeze(0), weight=weight_map.unsqueeze(0))
        return loss
