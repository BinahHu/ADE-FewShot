import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryMaskPredictor(nn.Module):
    def __init__(self, args):
        super(BinaryMaskPredictor, self).__init__()
        self.in_dim = args.feat_dim
        self.args = args
        self.down_sampling_rate = args.down_sampling_rate

        self.fc1 = nn.Conv2d(self.in_dim, 256, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def compute_anchor_location(anchor, scale, original_scale):
        anchor = np.array(anchor.detach().cpu())
        original_scale = np.array(original_scale.detach().cpu())
        anchor[:, 2] = anchor[:, 2] * scale[0] * original_scale[0]
        anchor[:, 3] = anchor[:, 3] * scale[0] * original_scale[0]
        anchor[:, 0] = anchor[:, 0] * scale[1] * original_scale[1]
        anchor[:, 1] = anchor[:, 1] * scale[1] * original_scale[1]
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
        anchor_num = labels.size

        # resize the mask into the same shape as feature map, avoid error in manual down sample computation
        scale_h = feature_map.shape[1] / mask.shape[0]
        scale_w = feature_map.shape[2] / mask.shape[1]
        mask = F.upsample_nearest(mask, size=(feature_map.shape[1], feature_map.shape[2]))
        anchors = self.compute_anchor_location(anchors, (scale_h, scale_w), scale)

        # enumerate the anchors and compute the loss
        loss = 0
        for i in range(anchor_num):
            anchor = anchors[i]
            label = labels[i]

            selected_feature_map = feature_map[:, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            pred_mask = self.fc2(F.relu(self.fc1(selected_feature_map)))
            tgt_mask = mask[anchor[2]:anchor[3], anchor[0]:anchor[1]].unsqueeze(0)
            # convert into 0-1 mask
            tgt_mask = torch.where(tgt_mask == label, 1.0, 0.0)

            loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
        return loss / (anchor_num + 1e-10)
