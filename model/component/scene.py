import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class SceneClassifier(nn.Module):
    def __init__(self, args):
        super(SceneClassifier, self).__init__()
        self.scene_num = -1
        self.in_dim = args.feat_dim
        self.feat_dim = -1
        for supervision in args.supervision:
            if supervision['name'] == 'scene':
                self.scene_num = supervision['other']['scene_num']
                self.pool_size = supervision['other']['pool_size']
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        self.fc = nn.Linear(self.pool_size * self.pool_size * self.in_dim, self.scene_num)
        self.loss = nn.CrossEntropyLoss()
        self.mode = 'train'

    def diagnosis(self, agg_data):
        x = agg_data['features']
        score = self.fc(x)
        return score

    def forward(self, agg_data):
        """
        forward pipeline, compute loss function
        :param agg_data: refer to ../base_model.py
        :return: loss
        """
        if self.mode == 'diagnosis':
            return self.diagnosis(agg_data)
        loss_sum = 0
        x = agg_data['feature_map']
        scene = agg_data['scene'].long()

        # Shallow supervision only
        x = self.pool(x.unsqueeze(0)).view(-1)
        label = scene
        score = self.fc(x)
        loss_sum += self.loss(score.unsqueeze(0), label.unsqueeze(0))
        return loss_sum
