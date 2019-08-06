# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity
import numpy as np



class GenericLoss(nn.Module):
    def __init__(self, feat_dim, is_soft, num_attr=0):
        super(GenericLoss, self).__init__()
        # self.attr_classifier = AttributeClassifier(160, 500).cuda()
        self.init_num = torch.cuda.current_device()
        """
        if is_soft:
            self.attr_loss = SoftConstraint(num_attr, feat_dim)
        else:
            self.attr_loss = HardConstrain(num_attr + 1, feat_dim)
        """
        self.is_soft = is_soft
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.num_attr = num_attr
        self.loss = nn.MultiLabelSoftMarginLoss()

    def forward(self, feats=None, embed=None, attributes=None, scores=None):
        if self.is_soft:
            labels = attributes.float().cuda()
            attr_loss = 0.0
            for i in range(len(labels)):
                nz = labels[i].nonzero()
                labels[i][nz] = 1.0

                loss_mask = torch.ones(attributes.size(1)).cuda()
                zeros = (labels[i, :] == 0).nonzero().cpu().numpy()
                indices = np.random.choice(zeros.squeeze(), int(round(len(zeros) * 0.8)), False)
                loss_mask[indices] = 0

                loss_fn = nn.MultiLabelSoftMarginLoss(weight=loss_mask)
                attr_loss += loss_fn(labels[i].unsqueeze(0), scores[i].unsqueeze(0))
            attr_loss /= labels.size(0)
        else:
            attr_loss = 1 - cosine_similarity(embed, feats, 1)
            attr_loss = attr_loss.mean()
        return attr_loss
