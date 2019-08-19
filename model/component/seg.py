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

        self.fc1 = nn.Conv2d(self.in_dim, 256, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

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

    def forward(self, agg_input):
        """
        take in the feature map and make predictions
        :param agg_input: input data
        :return: loss averaged over instances
        """
        feature_map = agg_input['feature_map']
        anchors = agg_input['anchors']
        scale = agg_input['scales']
        mask = agg_input['seg']
        labels = agg_input['labels']
        anchor_num = anchors.shape[0]

        feature_map = feature_map.unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)

        scale_h = feature_map.shape[2] / mask.shape[2]
        scale_w = feature_map.shape[3] / mask.shape[3]
        mask = F.interpolate(mask, size=(feature_map.shape[2], feature_map.shape[3]), mode='nearest')[0]
        anchors = self.compute_anchor_location(anchors, scale, (scale_h, scale_w))
        # enumerate the anchors and compute the loss
        loss = 0
        for i in range(anchor_num):
            anchor = anchors[i]
            label = labels[i]

            selected_feature_map = feature_map[:, :, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            # print(selected_feature_map.shape)
            pred_mask = self.fc2(F.relu(self.fc1(selected_feature_map)))[0]
            tgt_mask = mask[:, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            # convert into 0-1 mask
            ones = torch.ones(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            zeros = torch.zeros(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            tgt_mask = torch.where(tgt_mask == self.base_classes[int(label.item())], ones, zeros)
            loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
        return loss / (anchor_num + 1e-10)
