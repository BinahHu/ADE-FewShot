import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2


class BinaryMaskPredictor(nn.Module):
    def __init__(self, args):
        super(BinaryMaskPredictor, self).__init__()
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
        """
        visualize
        :param agg_input:
        :return:
        """
        def transform_mask_to_rgb(mask):
            mask = np.array(mask.cpu())
            R = ((mask / 256) * 10).astype(np.int8)
            G = (mask - 256 * (R/10)).astype(np.int)
            B = 125 * np.ones((mask.shape[0], mask.shape[1])).astype(np.int)
            mask_img = np.zeros((mask.shape[0], mask.shape[1], 3))
            mask_img[:, :, 0] = B
            mask_img[:, :, 1] = G
            mask_img[:, :, 2] = R
            mask_img = mask_img.astype(np.int)
            return mask_img

        feature_map = agg_input['feature_map']
        anchors = agg_input['anchors'].int()
        scale = agg_input['scales']
        mask = agg_input['seg']
        labels = agg_input['labels']
        anchor_num = anchors.shape[0]
        mask_img = transform_mask_to_rgb(mask)
        cv2.imwrite('diagnosis/data/mask/{}.jpg'.format(self.diagnosis_count), mask_img)

        feature_map = feature_map.unsqueeze(0)
        predicted_map = self.fc1(feature_map)
        predicted_map = F.interpolate(predicted_map, size=(mask.shape[0], mask.shape[1]))
        mask = mask.unsqueeze(0)
        anchors = self.compute_anchor_location(anchors, scale, [1, 1])

        # enumerate the anchors and compute the loss
        for i in range(anchor_num):
            anchor = anchors[i]
            label = labels[i]

            selected_map = predicted_map[:, :, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            pred_mask = F.sigmoid(5 * self.binary_transform(selected_map, label))

            tgt_mask = mask[:, anchor[2]:anchor[3], anchor[0]:anchor[1]]
            # convert into 0-1 mask
            ones = torch.ones(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            zeros = torch.zeros(tgt_mask.shape[0], tgt_mask.shape[1], tgt_mask.shape[2]).cuda()
            tgt_mask = torch.where(tgt_mask == self.base_classes[int(label.item())], ones, zeros)
            tgt_mask = np.array(255 * tgt_mask[0].cpu()).astype(np.int)
            cv2.imwrite('diagnosis/data/mask/{}_tgt_{}.jpg'.format(self.diagnosis_count, i), tgt_mask)
            # pred_mask = torch.where(pred_mask >= 0, ones, zeros)
            pred_mask = np.array(255 * pred_mask[0].detach().cpu()).astype(np.int)
            cv2.imwrite('diagnosis/data/mask/{}_pre_{}.jpg'.format(self.diagnosis_count, i), pred_mask)

        self.diagnosis_count += 1
        if self.diagnosis_count == 80:
            exit(0)
        return torch.tensor([0]).cuda()


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
