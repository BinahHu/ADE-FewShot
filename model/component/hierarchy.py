import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class HierarchyClassifier(nn.Module):
    def __init__(self, args):
        super(HierarchyClassifier, self).__init__()
        self.layer_width = []
        self.in_dim = args.feat_dim * args.crop_height * args.crop_width
        for supervision in args.supervision:
            if supervision['name'] == 'hierarchy':
                self.layer_width = supervision['other']['layer_width']
        self.fcs = nn.ModuleList()
        for width in self.layer_width:
            self.fcs.append(nn.Linear(self.in_dim, width))
        self.loss = nn.CrossEntropyLoss()
        self.mode = 'train'

    def diagnosis(self, agg_data):
        x = agg_data['features']
        scores = []
        for fc in self.fcs:
            scores.append(fc(x))
        return scores

    def forward(self, agg_data):
        """
        forward pipeline, compute loss function
        :param agg_data: refer to ../base_model.py
        :return: loss
        """
        if self.mode == 'diagnosis':
            return self.diagnosis(agg_data)
        loss_sum = 0
        x = agg_data['features']
        hierarchy = agg_data['hierarchy'].long()
        hierarchy = hierarchy[:x.shape[0]]

        # Shallow supervision only
        losses = []
        for i in range(len(self.fcs)):
            fc = self.fcs[i]
            label = hierarchy[:, i]
            score = fc(x)
            losses.append(self.loss(score, label))
        for loss in losses:
            loss_sum += loss
        return loss_sum