# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity
import json

attr_table = json.load(open("/home/zhu2/ADE-FewShot/data/ADE/ADE_Origin/attr.json"))


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
        if is_soft:
            num_attr += 1
        self.embedder = nn.Embedding(num_attr, feat_dim, padding_idx=0)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.num_attr = num_attr

    def forward(self, feats, embed):

        orth_loss = None
        attr_loss = None

        attr_loss = 1 - cosine_similarity(embed, feats, 1)
        attr_loss = attr_loss.mean()

        """
        if attributes.size(1) > 1:
            '''
            orth_loss = Variable(torch.zeros(1), requires_grad=True).cuda()
            for name, param in self.attr_loss.embedder.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat)).cuda()
                    sym -= Variable(torch.eye(param_flat.shape[0])).cuda()
                    orth_loss = orth_loss + sym.sum()

            orth_loss = orth_loss[0].abs()
            '''
            attr_loss = 0
            '''
            for i in range(len(feats)):
                attr_l, _ = self.attr_loss(attributes[i].cuda(), feats[i])
                attr_loss += attr_l
            '''
            attr_l, _ = self.attr_loss(attributes, feats)
            attr_loss += attr_l
        """
        return attr_loss, orth_loss
