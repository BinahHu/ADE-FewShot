import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class BinaryMaskPredictor(nn.Module):
    def __init__(self, args):
        super(BinaryMaskPredictor, self).__init__()
        self.in_dim = args.feat_dim
        self.args = args
        self.down_sampling_rate = args.down_sampling_rate

        self.fc1 = nn.Conv2d(self.in_dim, self.args.num_base_class + 1, kernel_size=3, stride=1, padding=1)
        # self.fc2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

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
        anchors = agg_input['anchors'].int()
        scale = agg_input['scales']
        mask = agg_input['seg']
        labels = agg_input['labels']
        anchor_num = anchors.shape[0]

        feature_map = feature_map.unsqueeze(0)
        predicted_map = self.fc1(feature_map)
        predicted_map = F.interpolate(predicted_map, size=(mask.shape[0], mask.shape[1]))

        mask = mask.unsqueeze(0)

        # enumerate the anchors and compute the loss
        loss = 0
        for i in range(anchor_num):
            anchor = anchors[i]
            label = labels[i]

            selected_map = predicted_map[:, :, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            pred_mask = self.binary_transform(selected_map, label)

            tgt_mask = mask[:, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            # convert into 0-1 mask
            ones = torch.ones(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            zeros = torch.zeros(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            tgt_mask = torch.where(tgt_mask == self.base_classes[int(label.item())], ones, zeros)
            loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
            print(loss)
        return loss / (anchor_num + 1e-10)
